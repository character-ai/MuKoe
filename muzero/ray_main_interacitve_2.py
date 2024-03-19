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
import argparse
import traceback
import reverb
from ray_reverb import RayReverbServer
import config

from ray_muzero_actor import RayMuzeroActor
from ray_learn import RayTpuLearner
from ray_inference import RayInferenceActor
from ray_inference_state import InferenceStateHandler
import math
from typing import Iterable, List

TPU_ID_REPR = "inference_v4_8_repr"
TPU_ID_DYNA = "inference_v4_8_dyna"
CPU_ID_REPR = "inference_repr_cpu_handler"
CPU_ID_DYNA = "inference_dyna_cpu_handler"

_GLOBAL_SUMMARY_FREQ_IN_S = 30
_REVERB_UPDATE_INTERVAL_IN_S = 300
_WEIGHT_UPDATE_FREQ_IN_S = 60
_LOOP_POLL_IN_S = 5

"""
python ray_main.py --ckpt_dir=/home/wendy/ray-train-test-pong/ckpt --tensorboard_dir=/home/wendy/ray-train-test-pong/tensorboard --reverb_dir=/home/wendy/ray-train-test-pong/reverb --num_actors=8 --environment=Pong
"""

parser = argparse.ArgumentParser(
    prog="Ray-RL-Demo", description="Our amazing MuZero implementation on Ray x TPUs!"
)
parser.add_argument(
    "--ckpt_dir",
    action="store",
    default="/home/wendy/ray-train-test-MsPacman-mukoe-6/ckpt",
)
parser.add_argument(
    "--save_dir",
    action="store",
    default="/home/wendy/ray-train-test-MsPacman-mukoe-6/log",
)
parser.add_argument(
    "--tensorboard_dir",
    action="store",
    default="/home/wendy/ray-train-test-MsPacman-mukoe-6/tensorboard",
)
parser.add_argument(
    "--reverb_dir",
    action="store",
    default="/home/wendy/ray-train-test-MsPacman-mukoe-6/reverb",
)

parser.add_argument("--num_actors", action="store", type=int, default=600)
# parser.add_argument("--core_per_task", action="store", type=int, default=14)
parser.add_argument("--core_per_task", action="store", type=int, default=12)
parser.add_argument("--environment", action="store", default="MsPacman")
parser.add_argument("--dyna_infer", action="store", default="cpu")
parser.add_argument("--repr_infer", action="store", default="tpu")

args = parser.parse_args()


def setup_loggers():
    logging.basicConfig(level=logging.INFO)


class MuzeroRunner:
    def __init__(self, args: argparse.ArgumentParser):
        self.args = args

        # Actor related structures
        self.actors = []
        self.actor_index_map = {}
        self.index_actor_map = {}

        # Learner related data structures
        self.learner_actor = None

        # Reverb related data structures
        self.reverb_actor = None
        self.reverb_future = None

        # Inference related structures
        self.tpu_inference_state = None
        self.tpu_inferers = []

        # Resiliency related structures
        self.futures_to_actors = {}
        self.actors_to_futures = {}
        self.futures = []

    def init_reverb(self):
        """Initializes all reverb related structures."""
        args = self.args
        logging.info("Initializing reverb server...")
        self.reverb_actor = RayReverbServer.remote(
            environment_name=args.environment, port=9090, reverb_dir=args.reverb_dir
        )
        # Initialize reverb actor and wait until ready.
        ray.get(self.reverb_actor.initialize.remote())
        self.reverb_server_address = ray.get(self.reverb_actor.get_ip.remote())
        logging.info("Reverb server address is: %s", self.reverb_server_address)
        self.reverb_client = reverb.Client(self.reverb_server_address)
        reverb_ckpt_path = self.reverb_client.checkpoint()
        logging.info("Reverb checkpoint server info: %s", reverb_ckpt_path)
        logging.info(self.reverb_client.server_info())
        print(self.reverb_client.server_info())
        self.reverb_future = self.reverb_actor.start.remote()

    def init_learner(self):
        """Initializes all learner related structures."""
        args = self.args
        logging.info("Initializing learner...")
        # set max_concurrency=2 to allow getting latest step
        self.learner_actor = RayTpuLearner.options(
            num_cpus=1,
            resources={"learner_cpu": 1},
            max_concurrency=2,
        ).remote(
            reverb_server_address=self.reverb_server_address,
            save_dir=args.save_dir,
            ckpt_dir=args.ckpt_dir,
            tensorboard_dir=args.tensorboard_dir,
            environment_name=args.environment,
        )
        learner_init_handle = self.learner_actor.initialize.remote()
        # learner has to be first initialized
        ray.get(learner_init_handle)

    def create_and_initialize_inferers(
        self, model_type: str, num_actors: int
    ) -> List[ray.actor.ActorHandle]:
        """Creates, initializes, and registers inference actors."""
        logging.info("Creating %dx %s TPU actors.", num_actors, model_type)
        inference_config = config.InferenceConfig
        inferer_handles = []
        inferer_ids = []

        if model_type == "repr":
            tpu_id_base = TPU_ID_REPR
            cpu_id_base = CPU_ID_REPR
            batch_size = inference_config.repr_batch_size
            batch_timeout_s = inference_config.repr_time_out
            weight_update_interval = inference_config.repr_update_interval
        else:
            tpu_id_base = TPU_ID_DYNA
            cpu_id_base = CPU_ID_DYNA
            batch_size = inference_config.dyna_batch_size
            batch_timeout_s = inference_config.dyna_time_out
            weight_update_interval = inference_config.dyna_update_interval

        # Create all actor handles
        for i in range(num_actors):
            inferer_id = self.tpu_inference_state.request_id(model=model_type)
            logging.info("Scheduling on actor %d.", inferer_id)
            # We set a concurrency of 2 to enable both the liveness check and batcher summary prints.
            inferer_handle = RayInferenceActor.options(
                num_cpus=1,
                resources={f"{cpu_id_base}_{inferer_id}": 1},
                max_concurrency=2,
            ).remote(
                ckpt_dir=self.args.ckpt_dir,
                batch_size=batch_size,
                batch_timeout_s=batch_timeout_s,
                model=model_type,
                tpu_id=f"{tpu_id_base}_{inferer_id}",
                weight_update_interval=weight_update_interval,
            )
            inferer_handles.append(inferer_handle)
            inferer_ids.append(inferer_id)

        # Initialize all actors.
        init_futures = [a.initialize.remote() for a in inferer_handles]
        ray.get(init_futures)

        # Grab endpoints.
        endpoint_futures = [a.get_endpoint.remote() for a in inferer_handles]
        endpoints = ray.get(endpoint_futures)

        # Register with the state manager.
        for inferer_handle, endpoint, inferer_id in zip(
            inferer_handles, endpoints, inferer_ids
        ):
            self.tpu_inference_state.register_inferer(
                model=model_type,
                actor=inferer_handle,
                endpoint=endpoint,
                inferer_id=inferer_id,
            )
        self.tpu_inferers += inferer_handles
        return inferer_handles

    def init_inference(self):
        """Initializes all inference related structures."""
        inference_config = config.InferenceConfig
        args = self.args
        if args.dyna_infer == "tpu":
            num_actors_per_dyna = inference_config.dyna_actor_per_replica
            num_dyna_inferers = math.ceil(args.num_actors / num_actors_per_dyna)
        else:
            num_dyna_inferers = 0
            num_actors_per_dyna = 0
        if args.repr_infer == "tpu":
            num_actors_per_repr = inference_config.repr_actor_per_replica
            num_repr_inferers = math.ceil(args.num_actors / num_actors_per_repr)
        else:
            num_repr_inferers = 0
            num_actors_per_repr = 0

        self.tpu_inference_state = InferenceStateHandler(
            repr_resource_base=TPU_ID_REPR,
            dyna_resource_base=TPU_ID_DYNA,
            num_dyna_inferers=num_dyna_inferers,
            num_repr_inferers=num_repr_inferers,
            num_actors_per_dyna=num_actors_per_dyna,
            num_actors_per_repr=num_actors_per_repr,
        )

        # Create and register dyna and repr inference actors
        if num_dyna_inferers > 0:
            self.create_and_initialize_inferers("dyna", num_dyna_inferers)
        if num_repr_inferers > 0:
            self.create_and_initialize_inferers("repr", num_repr_inferers)

    def init_muzero_actor(self, actor_i: int):
        args = self.args
        if actor_i % 100 == 0:
            action_i_save_dir = (
                ""  # this is the dir to save log, 
                    # TODO: add dir
            )
            action_i_tb_dir = args.tensorboard_dir
        else:
            action_i_save_dir = ""
            action_i_tb_dir = ""

        runtime_env_actor = {
            "env_vars": {
                "JAX_BACKEND": "CPU",
                "JAX_PLATFORMS": "cpu",
                "GCS_RESOLVE_REFRESH_SECS": "60",
                "RAY_memory_monitor_refresh_ms": "0",
            }
        }
        return RayMuzeroActor.options(
            num_cpus=args.core_per_task,
            runtime_env=runtime_env_actor,
            resources={"actor": 1},
        ).remote(
            reverb_server_address=self.reverb_server_address,
            environment_name=args.environment,
            save_dir=action_i_save_dir,
            is_evaluator=False,
            dyna_endpoint=self.tpu_inference_state.register_and_get_endpoint(
                model="dyna", actor_id=actor_i
            ),
            repr_endpoint=self.tpu_inference_state.register_and_get_endpoint(
                model="repr", actor_id=actor_i
            ),
            actor_id=actor_i,
            ckpt_dir=args.ckpt_dir,
            tensorboard_dir=action_i_tb_dir,
            use_priority_fn=False,
        )

    def init_actors(self, actor_ids: List[int]) -> List[ray.actor.ActorHandle]:
        """Initializes all actor related structures."""
        created_actors = []
        logging.info(
            "Instantiating %d actors using %d CPUs per actor...",
            len(actor_ids),
            self.args.core_per_task,
        )
        for actor_i in actor_ids:
            current_actor = self.init_muzero_actor(actor_i)
            self.actors.append(current_actor)
            self.actor_index_map[current_actor] = actor_i
            self.index_actor_map[actor_i] = current_actor
            created_actors.append(current_actor)

        logging.info("Initializing MuZero Actors...")
        actor_init_handles = [a.initialize.remote() for a in created_actors]
        logging.info("Waiting for initialization to finish...")
        ray.get(actor_init_handles)
        return created_actors

    def initialize(self):
        """Initializes all structures."""
        if "replay_buffer" not in ray.available_resources():
            raise ValueError(
                "While initializing, could not find a replay_buffer resource."
            )
        if "learner" not in ray.available_resources():
            raise ValueError("While initializing, could not find learner resources.")
        try:
            self.init_reverb()
            self.init_learner()
            self.init_inference()
            self.init_actors(actor_ids=range(self.args.num_actors))
            logging.info("All actors are initialized and ready to run.")
        except Exception:
            logging.info(
                "Caught error during actor init: %s. Shutting down",
                traceback.format_exc(),
            )
            ray.shutdown()
            exit(1)

    def start_reverb(self):
        """Registers reverb."""
        self.futures_to_actors[self.reverb_future] = self.reverb_actor
        self.actors_to_futures[self.reverb_actor] = self.reverb_future
        self.futures.append(self.reverb_future)

    def start_learner(self):
        """Starts and registers the learner."""
        future = self.learner_actor.train.remote()
        self.futures_to_actors[future] = self.learner_actor
        self.actors_to_futures[self.learner_actor] = future
        self.futures.append(future)

    def start_inference(self, inferers: Iterable[ray.actor.ActorHandle]):
        """Starts and registers inference servers."""
        for actor in inferers:
            future = actor.start.remote()
            self.futures_to_actors[future] = actor
            self.actors_to_futures[actor] = future
            self.futures.append(future)

    def start_actors(self, actors: Iterable[ray.actor.ActorHandle]):
        """Starts and registers actors."""
        for actor in actors:
            future = actor.run.remote()
            self.futures_to_actors[future] = actor
            self.actors_to_futures[actor] = future
            self.futures.append(future)

    def start(self):
        """Starts all components."""
        self.start_reverb()
        self.start_learner()
        self.start_inference(inferers=self.tpu_inferers)
        self.start_actors(actors=self.actors)

    def _stop_and_unregister_ray_actor(self, actor: ray.actor.ActorHandle):
        """Stops and unregisters an actor from resilience related structures."""
        future = self.actors_to_futures[actor]
        # Stop both the future and actor.
        ray.cancel(future)
        ray.kill(actor)
        del self.futures_to_actors[future]
        del self.actors_to_futures[actor]
        if future in self.futures:
            # Note this is automatically removed in the ray.wait() sequence.
            self.futures.remove(future)

    def stop_reverb(self):
        """Stops and unregisters the reverb server."""
        self._stop_and_unregister_ray_actor(self.reverb_actor)
        self.reverb_future = None

    def stop_learner(self):
        """Stops and unregisters the learner."""
        self._stop_and_unregister_ray_actor(self.learner_actor)

    def stop_inference(self, inferers: Iterable[ray.actor.ActorHandle]):
        """Stops and unregisters a set of inferers."""
        for inferer in inferers:
            self._stop_and_unregister_ray_actor(inferer)
            self.tpu_inferers.remove(inferer)
            self.tpu_inference_state.unregister_inferer(inferer)

    def stop_actors(self, actor_ids: Iterable[int]):
        """Stops and unregisters a set of actors."""
        for actor_i in actor_ids:
            actor = self.index_actor_map[actor_i]
            self._stop_and_unregister_ray_actor(actor)
            # Unregister from actor related structures.
            del self.index_actor_map[actor_i]
            del self.actor_index_map[actor]
            self.actors.remove(actor)

    def stop(self):
        """Stops all components.

        Note: This should only be used in drastic circumstances! I.e. if
        the learner or reverb servers fail.

        """
        logging.warning("Detected catastrophic error! Stopping all components.")
        self.stop_reverb()
        self.stop_learner()
        self.stop_inference(inferers=self.tpu_inferers)
        self.stop_actors(actor_ids=range(self.args.num_actors))

    def run_until_completion(self):
        self.start()
        results = []
        time_counter = 0
        latest_step = ray.get(self.learner_actor.get_latest_step.remote())
        print("The latest ckpt step is %d", latest_step)
        logging.info("The latest ckpt step is %d", latest_step)
        while self.futures:
            done_futures, self.futures = ray.wait(self.futures, timeout=_LOOP_POLL_IN_S)
            time_counter += 1
            actor_ids_to_restart = []
            dyna_inferers_to_restart = []
            repr_inferers_to_restart = []
            full_restart_needed = False

            # update reverb checkpoint
            if time_counter % (_REVERB_UPDATE_INTERVAL_IN_S // _LOOP_POLL_IN_S) == 0:
                reverb_ckpt_path = self.reverb_client.checkpoint()
                logging.info("Saving reverb to checkpoint: %s", reverb_ckpt_path)
                print("Saving reverb to checkpoint: %s", reverb_ckpt_path)

            # print the summary
            if time_counter % (_GLOBAL_SUMMARY_FREQ_IN_S // _LOOP_POLL_IN_S) == 0:
                # print the summary from inference actors with index 0
                for model in ["repr", "dyna"]:
                    inferer = self.tpu_inference_state.get_inference_handle(
                        model=model, actor_id=0
                    )
                    if inferer:
                        inferer.summarize_batcher_perf.remote()

            # update inference server weights
            if time_counter % (_WEIGHT_UPDATE_FREQ_IN_S // _LOOP_POLL_IN_S) == 0:
                new_latest_step = ray.get(self.learner_actor.get_latest_step.remote())
                if new_latest_step > latest_step:
                    print(
                        "Update inference servers to a new latest ckpt: %d",
                        new_latest_step,
                    )
                    logging.info(
                        "Update inference servers to a new latest ckpt: %d",
                        new_latest_step,
                    )
                    # update everything in the inference server
                    ray.get(
                        [server.update_weights.remote() for server in self.tpu_inferers]
                    )
                else:
                    print("No new ckpt. The current latest step: %d", new_latest_step)
                    logging.info(
                        "No new ckpt. The current latest step: %d", new_latest_step
                    )
                latest_step = new_latest_step

            for future in done_futures:
                try:
                    logging.info("Gathering failed futures")
                    results.append(ray.get(future))
                except ray.exceptions.RayActorError:
                    failed_actor = self.futures_to_actors[future]
                    logging.info("Actor %s failed.", failed_actor)
                    if (
                        failed_actor == self.learner_actor
                        or failed_actor == self.reverb_actor
                    ):
                        full_restart_needed = True
                    elif self.tpu_inference_state.model_type(failed_actor) is not None:
                        associated_actor_ids = (
                            self.tpu_inference_state.get_associated_actor_ids(
                                actor=failed_actor
                            )
                        )
                        actor_ids_to_restart.extend(associated_actor_ids)
                        model_type = self.tpu_inference_state.model_type(failed_actor)
                        logging.warning(
                            "Detected that a %s TPU inference actor has failed.",
                            model_type,
                        )
                        if model_type == "repr":
                            repr_inferers_to_restart.append(failed_actor)
                        else:
                            dyna_inferers_to_restart.append(failed_actor)
                    else:
                        actor_i = self.actor_index_map[failed_actor]
                        actor_ids_to_restart.append(actor_i)

            # Handle all failures in batch
            if full_restart_needed:
                logging.warning("Detected that the learner or reverb actor failed.")
                logging.warning("Restarting the entire run...")
                self.stop()
                self.initialize()
                self.start()
            else:
                logging.info("Detected %d actors to restart", len(actor_ids_to_restart))
                logging.info(
                    "Detected %d repr inferers to restart",
                    len(repr_inferers_to_restart),
                )
                logging.info(
                    "Detected %d dyna inferers to restart",
                    len(dyna_inferers_to_restart),
                )

                inferers = repr_inferers_to_restart + dyna_inferers_to_restart
                if inferers:
                    logging.info("Stopping inferers: %s.", inferers)
                    self.stop_inference(inferers=inferers)

                logging.info("Stopping actors: %s.", actor_ids_to_restart)
                self.stop_actors(actor_ids=actor_ids_to_restart)

                new_inferers = []
                if repr_inferers_to_restart:
                    logging.info(
                        "Re-creating %d repr inferers...", len(repr_inferers_to_restart)
                    )
                    new_inferers.extend(
                        self.create_and_initialize_inferers(
                            model_type="repr", num_actors=len(repr_inferers_to_restart)
                        )
                    )
                if dyna_inferers_to_restart:
                    logging.info(
                        "Re-creating %d dyna inferers...", len(dyna_inferers_to_restart)
                    )
                    new_inferers.extend(
                        self.create_and_initialize_inferers(
                            model_type="dyna", num_actors=len(dyna_inferers_to_restart)
                        )
                    )
                if new_inferers:
                    logging.info("Starting inferers.")
                    self.start_inference(inferers=new_inferers)

                logging.info("Re-starting actors: %s", actor_ids_to_restart)
                actors = self.init_actors(actor_ids=actor_ids_to_restart)
                self.start_actors(actors=actors)

                logging.info("Done with re-initializations!")


logging.info("Initializing the workload!")
logging.info("Input args: %s", args)
logging.info("Connecting to the Ray cluster.")

ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
logging.info("Available Ray resources: %s", ray.available_resources())
logging.info("All Ray resources: %s", ray.cluster_resources())

runner = MuzeroRunner(args)
runner.initialize()
logging.info("Starting the workload...")
runner.run_until_completion()
