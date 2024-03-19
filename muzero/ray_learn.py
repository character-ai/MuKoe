# Copyright 2024 Character Technologies Inc. and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import ray

import numpy as np
import time
from typing import Any, Tuple, Dict
import tensorflow as tf
import jax.numpy as jnp
import distutils.dir_util
import os
import jax
from jax._src.mesh import Mesh
from jax.sharding import PartitionSpec
from jax.experimental.pjit import pjit
import optax
import functools
import counting
from flax import core
from flax import struct
from flax.training import train_state as flax_train_state
from tensorboardX import SummaryWriter
from optax._src.transform import ScaleByAdamState, ScaleByScheduleState
import socket

import mcts.utils as mcts_utils
import adder
import atari_env_loop
import checkpoint_utils
import config
import losses
import networks
import optimizers
import reverb_dataset
import utils


class TrainState(flax_train_state.TrainState):
    """TrainState with an additional clamp_params field."""

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, step, params, target_params, tx, opt_state, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        return cls(
            step=step,
            apply_fn=apply_fn,
            params=params,  # here the params are pure params, not states
            target_params=target_params,  # here the params are pure params, not states
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def _step(
    data: reverb_dataset.ReverbData,
    train_state: TrainState,
    model: networks.MzNet,
    loss_params: config.LossConfig,
    optim_params: config.OptimConfig,
    num_unroll_steps: int,
    discount: float,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Performs one step graident update.
    Args:
      sample: reverb samples used to perform one step training.
      training_state: The existing states for training.
    Returns:
      A tuple of metrics and a new training_state after the training step.
    """
    gradients, metrics = jax.grad(
        losses.get_loss_and_metrics,
        has_aux=True,
        argnums=(5),
    )(
        model,
        loss_params,
        num_unroll_steps,
        discount,
        data,
        train_state.params,
        train_state.target_params,
        optim_params.weight_decay_type,
        optim_params.weight_decay_scale,
    )

    # the step, opt state and params are updated
    if optim_params.clip_by_global_norm > 0:
        clipping = optax.clip_by_global_norm(optim_params.clip_by_global_norm)
        gradients = clipping.update(gradients, None)[0]

    train_state = train_state.apply_gradients(grads=gradients)

    # TODO: also record the norm of updates
    for k, v in [("param", train_state.params), ("grad", gradients)]:
        compute_norm = jax.named_call(optax.global_norm, name=f"{k}_norm")
        metrics[f"loss/{k}_norm"] = compute_norm(v)

    # TODO: make update interval a hyperparameter
    target_params = jax.lax.cond(
        jnp.mod(train_state.step, 100) == 0,
        # jnp.mod(train_state.step, 500) == 0, # when running mspacman 0813 i used 500, not sure if its optimal
        lambda _: train_state.params,
        lambda _: train_state.target_params,
        operand=None,
    )

    new_train_state = train_state.replace(target_params=target_params)
    return new_train_state, metrics


@ray.remote
class RayLearnerShard:
    """Ray actor for a shard of the training loop.
    This intends to schedule only on a single TPU VM host.
    Attributes:
        reverb_server_address: The address of the Reverb server.
        save_dir: Directory where to save logs.
        ckpt_dir: Directory where to save checkpoints.
        tensorboard_dir: Directory where to save TensorBoard summaries.
        environment_name: The name of the Atari environment.
    """

    def __init__(
        self,
        shard_id: int,
        reverb_server_address: str,
        save_dir: str,
        ckpt_dir: str,
        tensorboard_dir: str,
        environment_name: str,
    ):
        self.shard_id = shard_id
        self.reverb_server_address = reverb_server_address
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.tensorboard_dir = tensorboard_dir
        self.environment_name = environment_name

    def live(self) -> bool:
        return True

    def initialize(self):
        try:
            print("Initializing RayLearnerShard. Environment ", os.environ)
            logging.info("Initializing RayLearnerShard. Environment: %s", os.environ)
            print("Shard id: ", self.shard_id)
            logging.debug("Shard id: %d", self.shard_id)
            logging.info("Initializing configs.")
            self.model_config = config.ModelConfig()
            self.optim_config = config.OptimConfig()
            self.train_config = config.TrainConfig()
            self.loss_config = config.LossConfig()
            self.mcts_config = mcts_utils.get_default_mcts_params()
            logging.info("Model config: %s", self.model_config)
            logging.info("Optim config: %s", self.optim_config)
            logging.info("Train config: %s", self.train_config)
            logging.info("Loss config: %s", self.loss_config)
            logging.info("MCTS config: %s", self.mcts_config)

            logging.info("Initializing dataset.")
            print("Model config: ", self.model_config)
            print("Optim config: ", self.optim_config)
            print("Train config: ", self.train_config)
            print("Loss config: ", self.loss_config)
            print("MCTS config: ", self.mcts_config)
            print("Initializing dataset.")
            dataset = reverb_dataset.make_reverb_dataset(
                server_address=self.reverb_server_address,
                batch_size=None,
                prefetch_size=None,
                max_in_flight_samples_per_worker=2 * self.train_config.batchsize,
                table=adder.DEFAULT_PRIORITY_TABLE,
            )
            dataset = reverb_dataset.make_mz_dataset(
                dataset,
                self.train_config.batchsize,
                tf.data.AUTOTUNE,
                self.train_config.td_steps,
                use_raw_value=self.loss_config.use_raw_value,
                discount=self.mcts_config.value_head["discount"],
            )
            self.dataset = reverb_dataset.NumpyIterator(dataset)
            print("Dataset created")
            # TODO: make loader more efficient
            # self._iterator = utils.sharded_prefetch(
            #    self._iterator,
            #    devices=local_devices,
            #    num_threads=len(local_devices),
            # )
            # dataset = reverb_dataset.make_mz_dataset(server_address=reverb_server_address,
            #                batch_size=train_config.batchsize,
            #                sequence_length=train_config.num_unroll_steps,
            #                table=adder.DEFAULT_PRIORITY_TABLE,
            #                num_parallel_calls=tf.data.AUTOTUNE,
            #                )

            _, self.key2 = jax.random.split(jax.random.PRNGKey(42))
            print("Key created")
            atari_env_loop.make_atari_environment(self.environment_name)
            print("environment created")
            input_specs = atari_env_loop.get_environment_spec_for_init_network(
                self.environment_name
            )
            self.model = networks.get_model(self.model_config)
            print("model created")
            # dummy inputs
            dummy_obs = utils.add_batch_dim(utils.zeros_like(input_specs))
            dummy_obs = utils.add_batch_size(dummy_obs, self.train_config.batchsize)
            self.dummy_obs = dummy_obs.observations
            print("dummy obs created")
            # make optimizer
            self.get_learning_rate_schedule = functools.partial(
                optimizers.get_learning_rate_schedule,
                config=self.optim_config,
                total_training_steps=self.train_config.total_training_steps,
            )
            print("lr created")
            optimizer = optimizers.make_optimizer(self.optim_config)
            self.optimizer = optax.chain(
                optimizer,
                optax.scale_by_schedule(self.get_learning_rate_schedule),
                optax.scale(-1),
            )

            logging.info("Creating ckpt manager.")
            print("Creating ckpt manager")
            self.ckpt_manager = checkpoint_utils.get_ckpt_manager(
                self.ckpt_dir, self.train_config.ckpt_save_interval_steps
            )

            if self.shard_id == 0:
                self.is_chief = True
                self.writer = SummaryWriter(self.tensorboard_dir + "/train")
            else:
                self.is_chief = False
                self.writer = None

            self.counter = None
            self.counter = counting.Counter(self.counter, "learner")

            logging.info(
                f"To see full metrics 'tensorboard --logdir={self.tensorboard_dir}'"
            )

            self.mesh = Mesh(np.asarray(jax.devices(), dtype=object), ["data"])
            dummy_action = jnp.zeros((self.train_config.batchsize, 1), dtype=jnp.int32)

            all_steps = self.ckpt_manager.all_steps(read=True)
            latest_step = max(all_steps) if all_steps else None
            with self.mesh:
                if latest_step is None:
                    model_vars = pjit(
                        self.model.init,
                        in_shardings=(
                            None,
                            PartitionSpec("data"),
                            PartitionSpec("data"),
                        ),
                        out_shardings=None,
                    )(self.key2, self.dummy_obs, dummy_action)
                    target_vars = model_vars.copy({})
                    opt_state = pjit(
                        self.optimizer.init,
                        in_shardings=None,
                        out_shardings=None,
                    )(model_vars["params"])
                    self.train_state = TrainState.create(
                        apply_fn=self.model.apply,
                        params=model_vars["params"],
                        target_params=target_vars["params"],
                        step=0,
                        tx=self.optimizer,
                        opt_state=opt_state,
                    )
                    state = {"state": self.train_state}
                    self.ckpt_manager.save(0, {"save_state": state})
                    self.ckpt_manager.wait_until_finished()
                    logging.info("done saving")
                else:
                    logging.info(f"loading model at step {latest_step}")
                    restored = self.ckpt_manager.restore(latest_step)
                    restored_state = restored["save_state"]["state"]
                    opt_state = pjit(
                        self.optimizer.init, in_shardings=None, out_shardings=None
                    )(restored_state["params"])
                    restored_opt_state = restored_state["opt_state"]
                    """
                    ((EmptyState(), ScaleByAdamState(count=Array(0, dtype=int32), mu, nu), 
                    EmptyState(), 
                    AddWeightDecayState()), 
                    ScaleByScheduleState(count=Array(0, dtype=int32)), 
                    EmptyState())
                    """
                    new_ScaleByAdamState = ScaleByAdamState(
                        count=restored_opt_state[0][1]["count"],
                        mu=restored_opt_state[0][1]["mu"],
                        nu=restored_opt_state[0][1]["nu"],
                    )
                    new_ScaleByScheduleState = ScaleByScheduleState(
                        count=restored_opt_state[1]["count"]
                    )
                    opt_state = (
                        (
                            opt_state[0][0],
                            new_ScaleByAdamState,
                            opt_state[0][2],
                            opt_state[0][3],
                        ),
                        new_ScaleByScheduleState,
                        opt_state[2],
                    )
                    self.train_state = TrainState.create(
                        apply_fn=self.model.apply,
                        params=restored_state["params"],
                        target_params=restored_state["target_params"],
                        step=restored_state["step"],
                        tx=self.optimizer,
                        opt_state=opt_state,
                    )

            if latest_step is None:
                self.starting_step = 0
                self.latest_step = 0
            else:
                self.starting_step = latest_step
                self.latest_step = latest_step
            print(f"Done init learner at step {self.starting_step}")
            logging.info(f"Done init learner at step {self.starting_step}")
        except Exception as e:
            print("process_batch failed due to: ", e)
            raise e

    def train(self):
        partial_step = functools.partial(
            _step,
            model=self.model,
            loss_params=self.loss_config,
            optim_params=self.optim_config,
            num_unroll_steps=self.train_config.num_unroll_steps,
            discount=self.mcts_config.value_head["discount"],
        )
        p_train_step = pjit(
            partial_step,
            in_shardings=(PartitionSpec("data"), None),
            out_shardings=(None, None),
        )

        logging.info("starting to train")
        timestamp = time.time()
        for train_step in range(
            self.starting_step, self.train_config.total_training_steps
        ):
            logging.info("train_step=%d", train_step)
            logging.info("starting to get next dataset")
            start_reverb = time.time()
            sample = next(self.dataset)
            end_reverb = time.time()
            reverb_diff = end_reverb - start_reverb
            logging.info(f"sampling rever takes {reverb_diff}")
            logging.info("global batch shape")

            global_batch = sample.data
            global_data_shape = jax.tree_util.tree_map(lambda x: x.shape, global_batch)
            logging.info(global_data_shape)
            local_batch = jax.tree_util.tree_map(
                lambda x: x[jax.process_index() :: jax.process_count()], global_batch
            )
            local_data_shape = jax.tree_util.tree_map(lambda x: x.shape, local_batch)
            logging.info("local data shape")
            logging.info(local_data_shape)
            batch = reverb_dataset.get_next_batch_sharded(
                local_batch, PartitionSpec("data"), global_data_shape, self.mesh
            )

            with self.mesh:
                self.train_state, results = p_train_step(batch, self.train_state)
                del results["debug_value_final"]
            with jax.spmd_mode("allow_all"):
                # Update our counts and record from the chief learner.
                if self.is_chief:
                    logging.info("chief learner logging starts")
                    # Compute elapsed time.
                    new_timestamp = time.time()
                    elapsed_time = new_timestamp - timestamp
                    timestamp = new_timestamp
                    results["step/sec"] = 1 / elapsed_time if elapsed_time != 0 else 0
                    results["reverb_time"] = end_reverb - start_reverb

                    # opt_state = ((optax._src.transform.TraceState,
                    #               optax._src.base.EmptyState),
                    #              optax._src.transform.ScaleByScheduleState,
                    #              optax._src.base.EmptyState)
                    # Since the optimizer keeps track of its own learner step count, we need to
                    # use this when plotting the learning rate.
                    optimizer_learner_steps = self.train_state.opt_state[0][1].count

                    results["optimizer_learner_steps"] = float(optimizer_learner_steps)
                    results["learning_rate"] = float(
                        self.get_learning_rate_schedule(optimizer_learner_steps)
                    )

                    counts = self.counter.increment(steps=1, walltime=elapsed_time)
                    # this learner steps is only used to calculate the step/sec
                    learner_steps = counts["learner_steps"]

                    results["network_steps_age"] = train_step - jnp.mean(
                        sample.data.extras[networks.NETWORK_STEPS]
                    )

                    if counts["learner_walltime"] == 0:
                        results["avg_step/sec"] = 0
                    else:
                        results["avg_step/sec"] = (
                            learner_steps / counts["learner_walltime"]
                        )

                    # learnter_steps in counts is the step key used by the learner and actors
                    # in tensorboard.
                    results.update(counts)
                    results["state_learner_steps"] = self.train_state.step
                    utils.write_metrics(
                        self.writer, results, train_step, self.train_config.log_period
                    )

            save_state = {
                "state": self.train_state,
            }
            try:
                self.ckpt_manager.save(train_step, {"save_state": save_state})
                self.ckpt_manager.wait_until_finished()
            except Exception:
                logging.info("saving failed")

            all_steps = self.ckpt_manager.all_steps(read=True)
            latest_step = max(all_steps) if all_steps else None
            logging.info(f"learner_saved_latest_step={latest_step}")
            self.latest_step = latest_step
            if train_step % self.train_config.log_period == 0:
                logging.info(results)

        if self.save_dir != "":
            print("learner copying tmp log to persistent nsf")
            distutils.dir_util.copy_tree(
                "/tmp/ray/session_latest/logs", os.path.join(self.save_dir, "learner")
            )

    def __repr__(self):
        return f"[RayLearnerShard-w{self.shard_id}]"

    def get_latest_step(self):
        return self.latest_step


@ray.remote
class RayTpuLearner:
    """Global manager for all learner shards.
    Attributes:
        reverb_server_address: The address of the Reverb server.
        save_dir: Directory where to save logs.
        ckpt_dir: Directory where to save checkpoints.
        tensorboard_dir: Directory where to save TensorBoard summaries.
        environment_name: The name of the Atari environment.
    """

    def __init__(
        self,
        reverb_server_address: str,
        save_dir: str,
        ckpt_dir: str,
        tensorboard_dir: str,
        environment_name: str,
    ):
        logging.info("Initializing RayTpuLearner")
        tpu_id = "learner"
        num_learners = int(ray.available_resources()[tpu_id])
        logging.info("Number of detected learners: %d", num_learners)
        print("Number of detected learners: ", num_learners)
        self._learners = [
            # set max_concurrency=2 to allow getting latest step
            RayLearnerShard.options(
                resources={"TPU": 4, tpu_id: 1}, max_concurrency=2
            ).remote(
                shard_id=i,
                reverb_server_address=reverb_server_address,
                save_dir=save_dir,
                ckpt_dir=ckpt_dir,
                tensorboard_dir=tensorboard_dir,
                environment_name=environment_name,
            )
            for i in range(num_learners)
        ]

    def initialize(self):
        logging.info("Initializing...")
        return ray.get([t.initialize.remote() for t in self._learners])

    def train(self):
        # Do we want to return anything?
        logging.info("Starting to train")
        return ray.get([t.train.remote() for t in self._learners])

    def get_latest_step(self):
        chief_learner = self._learners[0]
        return ray.get(chief_learner.get_latest_step.remote())

    def __repr__(self):
        return f"[RayTpuLearner]({socket.gethostname()})"
