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
import reverb
import distutils.dir_util
import os
import jax
import adder
import atari_env_loop
import config
import networks
import specs
import utils
from actor import MzActor
from typing import Optional
import socket

jax.config.update("jax_platform_name", "cpu")


# TODO - merge MzActor with RayMuzeroActor
@ray.remote(num_cpus=1, resources={"actor": 1})
class RayMuzeroActor:
    """Ray actor wrapper for the Muzero actor.
    Attributes:
        reverb_server_address: The address of the Reverb server.
        environment_name: The name of the Atari environment
        save_dir: The directory to save logs.
        is_evaluator: Whether or not this actor is being used for evaluation.
        actor_id: The index of the actor.
        ckpt_dir: The directory where to save checkpoints.
        tensorboard_dir: The directory where to save tensorboard summaries.
        use_priority_fn: Whether or not to use the priority function.
        dyna_endpoint: If provided, this represents the endpoint of the associated
          TPU server for dyna inferences.
        repr_endpoint: If provided, this represents the endpoint of the associated
          TPU server for repr inferences.
    """

    def __init__(
        self,
        reverb_server_address: str,
        environment_name: str,
        save_dir: str,
        is_evaluator: bool,
        actor_id: int,
        ckpt_dir: str,
        tensorboard_dir: str,
        use_priority_fn: bool,
        dyna_endpoint: Optional[str] = None,
        repr_endpoint: Optional[str] = None,
    ):
        logging.info("OS Environment: %s", os.environ)
        self.reverb_server_address = reverb_server_address
        self.environment_name = environment_name
        self.save_dir = save_dir
        self.is_evaluator = is_evaluator
        self.actor_id = actor_id
        self.ckpt_dir = ckpt_dir
        self.tensorboard_dir = tensorboard_dir
        self.use_priority_fn = use_priority_fn

        self.dyna_endpoint = dyna_endpoint
        self.repr_endpoint = repr_endpoint

    def initialize(self):
        model_config = config.ModelConfig()
        train_config = config.TrainConfig()
        replay_config = config.ReplayConfig()

        key1, _ = jax.random.split(jax.random.PRNGKey(self.actor_id))

        environment = atari_env_loop.make_atari_environment(self.environment_name)
        environment_specs = specs.make_environment_spec(environment)
        input_specs = atari_env_loop.get_environment_spec_for_init_network(
            self.environment_name
        )
        extra_specs = atari_env_loop.get_extra_spec()

        model = networks.get_model(model_config)

        _unroll_step = max(train_config.td_steps, train_config.num_unroll_steps)
        reverb_client = reverb.Client(self.reverb_server_address)

        if self.use_priority_fn:
            priority_fn = utils.compute_td_priority
        else:
            priority_fn = None

        atari_adder = adder.SequenceAdder(
            client=reverb_client,
            # the period should be shorter than replay seq length in order to
            # include all possible starting points. Particularly the margin should
            # be at least stack_frame + unroll + 1 to include everything
            # to be safe we also * 2 here, but in the most rigorous way, we should
            # not * 2.
            # TODO: test not to * 2
            period=replay_config.replay_sequence_length
            - (atari_env_loop.ATARI_NUMBER_STACK_FRAME + _unroll_step + 1) * 2,
            sequence_length=replay_config.replay_sequence_length,
            end_of_episode_behavior=adder.EndBehavior.WRITE,
            environment_spec=environment_specs,
            extras_spec=extra_specs,
            init_padding=atari_env_loop.ATARI_NUMBER_STACK_FRAME - 1,
            end_padding=_unroll_step,
            priority_fns={adder.DEFAULT_PRIORITY_TABLE: priority_fn},
        )

        mz_actor = MzActor(
            network=model,
            observation_spec=input_specs,
            actor_id=self.actor_id,
            rng=key1,
            ckpt_dir=self.ckpt_dir,
            ckpt_save_interval_steps=train_config.ckpt_save_interval_steps,
            adder=atari_adder,
            mcts_params=None,  # TODO: change!
            use_argmax=False,
            use_mcts=True,
            dyna_endpoint=self.dyna_endpoint,
            repr_endpoint=self.repr_endpoint,
            total_training_steps=train_config.total_training_steps,
        )
        if self.actor_id == 0:
            reverb_client = self.reverb_server_address
        else:
            reverb_client = ""

        self.env_loop = atari_env_loop.AtariEnvironmentLoop(
            environment=environment,
            actor=mz_actor,
            reverb_client=reverb_client,
            tensorboard_dir=self.tensorboard_dir,
        )

    def run(self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None):
        logging.info("Running the environment loop.")
        self.env_loop.run(num_episodes=num_episodes, num_steps=num_steps)
        if self.save_dir != "":
            print("copying tmp log to persistent nsf")
            distutils.dir_util.copy_tree(
                "/tmp/ray/session_latest/logs", os.path.join(self.save_dir, "act")
            )

    def __repr__(self):
        """Prints out formatted Ray actor logs."""
        return f"[MuZeroActor: {self.actor_id}]({socket.gethostname()})"
