# Copyright 2024 The MuKoe Authors
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
"""A simple agent-environment loop."""

import time
from typing import Optional, Tuple
import functools

from absl import logging
from env_wrapper import (
    GymAtariAdapter,
    SinglePrecisionWrapper,
    AtariWrapper,
    wrap_all,
    FrameStacker,
)
import numpy as np
import dm_env
import gym
import specs
import networks
import counting

# from acme.utils import loggers
from actor import MzActor
import utils

# create an atari environment
ATARI_NUMBER_STACK_FRAME = 4


def make_atari_environment(
    level: str = "Pong",
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    flatten_frame_stack: bool = False,
    grayscaling: bool = False,
    to_float: bool = False,
    scale_dims: Tuple[int, int] = (96, 96),
) -> dm_env.Environment:
    """Loads the Atari environment."""
    # Internal logic.
    version = "v0" if sticky_actions else "v4"
    level_name = f"{level}NoFrameskip-{version}"
    env = gym.make(level_name, full_action_space=True)

    wrapper_list = [
        GymAtariAdapter,
        functools.partial(
            AtariWrapper,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=108_000,
            num_stacked_frames=1,  # Always ONLY use 1 frame
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            zero_discount_on_life_loss=zero_discount_on_life_loss,
        ),
        SinglePrecisionWrapper,
    ]

    return wrap_all(env, wrapper_list)


def get_environment_spec_for_init_network(
    level: str = "Pong",
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    flatten_frame_stack: bool = False,
    grayscaling: bool = False,
    to_float: bool = False,
    scale_dims: Tuple[int, int] = (96, 96),
) -> specs.EnvironmentSpec:
    """Construct atari env."""
    version = "v0" if sticky_actions else "v4"
    level_name = f"{level}NoFrameskip-{version}"
    env = gym.make(level_name, full_action_space=True)

    wrapper_list = [
        GymAtariAdapter,
        functools.partial(
            AtariWrapper,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=108_000,
            num_stacked_frames=ATARI_NUMBER_STACK_FRAME,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            zero_discount_on_life_loss=zero_discount_on_life_loss,
        ),
        SinglePrecisionWrapper,
    ]

    env = wrap_all(env, wrapper_list)

    return specs.make_environment_spec(env)


def get_extra_spec(discrete_action_space: int = 18, use_mcts: bool = True):
    # record policy_probs in the replay buffer.
    return {
        networks.POLICY_PROBS: specs.Array(
            shape=(discrete_action_space,), dtype="float32"
        ),
        networks.NETWORK_STEPS: specs.Array(shape=(), dtype="int32"),
        networks.RAW_VALUE: specs.Array(shape=(), dtype="float32"),
    }


class AtariEnvironmentLoop:
    """A simple environment loop specialized for L2E.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. This can be used as:

      loop = AtariEnvironmentLoop(environment, actor)
      loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        actor: MzActor,
        reverb_client: str = "",
        counter: Optional[counting.Counter] = None,
        inference_network_update_interval: int = 5000,
        is_evaluator: bool = False,
        label: str = "atari_environment_loop",
        tensorboard_dir: str = "",
        # logger: Optional[loggers.Logger] = None,
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        # self._logger = logger or loggers.make_default_logger(label)
        self.inference_network_update_interval = inference_network_update_interval
        self.is_evaluator = is_evaluator
        self._frame_stacker = FrameStacker(num_frames=ATARI_NUMBER_STACK_FRAME)
        if reverb_client == "":
            self._reverb_client = None
        else:
            import reverb

            self._reverb_client = reverb.Client(reverb_client)

        if tensorboard_dir == "":
            self._writer = None
        else:
            from tensorboardX import SummaryWriter

            self._writer = SummaryWriter(tensorboard_dir + "/act")

    def run(self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop for `num_episodes` episodes. Each episode is itself
        a loop which interacts first with the environment to get an observation and
        then give that observation to the agent in order to retrieve an action. Upon
        termination of an episode a new episode will be started. If the number of
        episodes is not given then this will interact with the environment
        infinitely.

        Args:
          num_episodes: number of episodes to run the loop for. If `None` (default),
            runs without limit.
          num_steps: number of steps to run the loop for. If `None` (default),
            runs without limit.
        """

        # if not (num_episodes is None or num_steps is None):
        # raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            if num_episodes is None and num_steps is None:
                return False
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count, step_count = 0, 0
        while (
            not should_terminate(episode_count, step_count)
            and not self._actor.finished()
        ):
            # Reset any counts and start the environment.
            start_time = time.time()
            episode_steps = 0
            episode_return = 0
            self._frame_stacker.reset()
            timestep = self._environment.reset()
            env_reset_latency = time.time() - start_time

            stack_obs = self._frame_stacker.step(timestep.observation)
            stack_obs = stack_obs[:, :, :, 0, :]  # (96, 96, 3, 1, 4)
            logging.info("starting this episode")
            self._actor.observe_first(timestep)

            # Run an episode.
            acting_step = self._actor._latest_step
            select_action_durations = []
            env_step_durations = []
            actor_metrics = {}
            while not timestep.last():
                # Generate an action from the agent's policy and step the environment.
                action_start = time.time()
                # make actor carry train/eval info
                if self.is_evaluator:
                    action = self._actor.select_action(stack_obs, is_training=False)
                else:
                    action = self._actor.select_action(stack_obs)
                select_action_durations.append(time.time() - action_start)
                if hasattr(self._actor, "get_metrics") and callable(
                    self._actor.get_metrics
                ):
                    for key, value in self._actor.get_metrics().items():
                        if key not in actor_metrics:
                            actor_metrics[key] = []
                        actor_metrics[key].append(value)
                env_start = time.time()
                timestep = self._environment.step(action)

                stack_obs = self._frame_stacker.step(timestep.observation)
                stack_obs = stack_obs[:, :, :, 0, :]  # (96, 96, 3, 1, 4)

                env_step_durations.append(time.time() - env_start)

                # Have the agent observe the timestep.
                self._actor.observe(action, next_timestep=timestep)

                # Book-keeping.
                episode_steps += 1
                episode_return += timestep.reward
                if episode_steps % self.inference_network_update_interval == 0:
                    self._actor.update()

            # Let the actor update itself.
            begin_update = time.time()
            self._actor.update()
            end_update = time.time()
            logging.info(f"Note: It takes {end_update - begin_update} to update actor")

            if self._reverb_client is not None:
                logging.info("Reverb Info:")
                logging.info(self._reverb_client.server_info())

            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

            # Collect the results and combine with counts.
            episode_duration = time.time() - start_time
            steps_per_second = episode_steps / episode_duration
            result = {
                "act/episode_length": episode_steps,
                "act/episode_return": episode_return,
                "act/episode_duration": episode_duration,
                "act/steps_per_second": steps_per_second,
                "act/select_action_latency_sec": np.mean(select_action_durations),
                "act/env_reset_latency_sec": env_reset_latency,
                "act/env_step_latency_sec": np.mean(env_step_durations),
                "act/acting_network_step": acting_step,
            }
            result.update(counts)
            for key, values in actor_metrics.items():
                result[key] = np.mean(values)

            step_count += episode_steps
            episode_count += 1
            # Log the given results.
            if self._writer is None:
                logging.info(result)
            else:
                utils.write_metrics(self._writer, result, step_count, 1)

    def generate_episode(self):
        """Generate a single eval episode.

        The action is generated by greedily using MCTS policy
        """
        observations_all = []
        episode_return = 0
        self._frame_stacker.reset()
        timestep = self._environment.reset()

        observations_all.append(timestep.observation)
        stack_obs = self._frame_stacker.step(timestep.observation)
        stack_obs = stack_obs[:, :, :, 0, :]  # (96, 96, 3, 1, 4)
        logging.info("starting this episode")
        self._actor.observe_first(timestep)

        # Run an episode.
        while not timestep.last():
            # Generate an action greedily from the agent's mcts policy
            action = self._actor.select_action(stack_obs, is_training=False)
            # Step the environment.
            timestep = self._environment.step(action)
            observations_all.append(timestep.observation)
            stack_obs = self._frame_stacker.step(timestep.observation)
            stack_obs = stack_obs[:, :, :, 0, :]  # (96, 96, 3, 1, 4)

            # Have the agent observe the timestep.
            self._actor.observe(action, next_timestep=timestep)

            # Book-keeping.
            logging.info(f"reward {timestep.reward}")
            print(f"reward {timestep.reward}")
            episode_return += timestep.reward

        return episode_return, observations_all
