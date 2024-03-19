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
import enum
import abc
import operator
from typing import (
    Optional,
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Sized,
    Union,
    Tuple,
    Dict,
)

import specs
import acme_types as types
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import tensorflow as tf
import tree
import logging
import time

DEFAULT_PRIORITY_TABLE = "priority_table"
_MIN_WRITER_LIFESPAN_SECONDS = 60
StartOfEpisodeType = Union[bool, specs.Array, tf.Tensor, tf.TensorSpec, Tuple[()]]


class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observation: types.NestedArray
    action: types.NestedArray
    reward: types.NestedArray
    discount: types.NestedArray
    start_of_episode: StartOfEpisodeType
    extras: types.NestedArray = ()


Trajectory = Step


class PriorityFnInput(NamedTuple):
    """The input to a priority function consisting of stacked steps."""

    observations: types.NestedArray
    actions: types.NestedArray
    rewards: types.NestedArray
    discounts: types.NestedArray
    start_of_episode: types.NestedArray
    extras: types.NestedArray


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]


def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


def calculate_priorities(
    priority_fns: PriorityFnMapping,
    trajectory_or_transition: Union[Trajectory, types.Transition],
) -> Dict[str, float]:
    """Helper used to calculate the priority of a Trajectory or Transition.

    This helper converts the leaves of the Trajectory or Transition from
    `reverb.TrajectoryColumn` objects into numpy arrays. The converted Trajectory
    or Transition is then passed into each of the functions in `priority_fns`.

    Args:
      priority_fns: a mapping from table names to priority functions (i.e. a
        callable of type PriorityFn). The given function will be used to generate
        the priority (a float) for the given table.
      trajectory_or_transition: the trajectory or transition used to compute
        priorities.

    Returns:
      A dictionary mapping from table names to the priority (a float) for the
      given collection Trajectory or Transition.
    """
    if any([priority_fn is not None for priority_fn in priority_fns.values()]):
        trajectory_or_transition = tree.map_structure(
            lambda col: col.numpy(), trajectory_or_transition
        )

    return {
        table: (priority_fn(trajectory_or_transition) if priority_fn else 1.0)
        for table, priority_fn in priority_fns.items()
    }


class Adder(abc.ABC):
    """The Adder interface.

    An adder packs together data to send to the replay buffer, and potentially
    performs some reduction/transformation to this data in the process.

    All adders will use this API. Below is an illustrative example of how they
    are intended to be used in a typical RL run-loop. We assume that the
    environment conforms to the dm_env environment API.

    ```python
    # Reset the environment and add the first observation.
    timestep = env.reset()
    adder.add_first(timestep.observation)

    while not timestep.last():
      # Generate an action from the policy and step the environment.
      action = my_policy(timestep)
      timestep = env.step(action)

      # Add the action and the resulting timestep.
      adder.add(action, next_timestep=timestep)
    ```

    Note that for all adders, the `add()` method expects an action taken and the
    *resulting* timestep observed after taking this action. Note that this
    timestep is named `next_timestep` precisely to emphasize this point.
    """

    @abc.abstractmethod
    def add_first(self, timestep: dm_env.TimeStep):
        """Defines the interface for an adder's `add_first` method.

        We expect this to be called at the beginning of each episode and it will
        start a trajectory to be added to replay with an initial observation.

        Args:
          timestep: a dm_env TimeStep corresponding to the first step.
        """

    @abc.abstractmethod
    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ):
        """Defines the adder `add` interface.

        Args:
          action: A possibly nested structure corresponding to a_t.
          next_timestep: A dm_env Timestep object corresponding to the resulting
            data obtained by taking the given action.
          extras: A possibly nested structure of extra data to add to replay.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the adder's buffer."""


class ReverbAdder(Adder):
    """Base class for Reverb adders."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        max_in_flight_items: int,
        delta_encoded: bool = False,
        priority_fns: Optional[PriorityFnMapping] = None,
        validate_items: bool = True,
    ):
        """Initialize a ReverbAdder instance.

        Args:
          client: A client to the Reverb backend.
          max_sequence_length: The maximum length of sequences (corresponding to the
            number of observations) that can be added to replay.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          priority_fns: A mapping from table names to priority functions; if
            omitted, all transitions/steps/sequences are given uniform priorities
            (1.0) and placed in DEFAULT_PRIORITY_TABLE.
          validate_items: Whether to validate items against the table signature
            before they are sent to the server. This requires table signature to be
            fetched from the server and cached locally.
        """
        if priority_fns:
            priority_fns = dict(priority_fns)
        else:
            priority_fns = {DEFAULT_PRIORITY_TABLE: None}

        self._client = client
        self._priority_fns = priority_fns
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        self._max_in_flight_items = max_in_flight_items
        self._add_first_called = False

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None
        # Every time a new writer is created, it must fetch the signature from the
        # Reverb server. If this is set too low it can crash the adders in a
        # distributed setup where the replay may take a while to spin up.
        self._validate_items = validate_items

    def __del__(self):
        if self.__writer is not None:
            timeout_ms = 10_000
            # Try flush all appended data before closing to avoid loss of experience.
            try:
                self.__writer.flush(0, timeout_ms=timeout_ms)
            except reverb.DeadlineExceededError as e:
                logging.error(
                    "Timeout (%d ms) exceeded when flushing the writer before "
                    "deleting it. Caught Reverb exception: %s",
                    timeout_ms,
                    str(e),
                )
            self.__writer.close()
            self.__writer = None

    @property
    def _writer(self) -> reverb.TrajectoryWriter:
        if self.__writer is None:
            self.__writer = self._client.trajectory_writer(
                num_keep_alive_refs=self._max_sequence_length,
                validate_items=self._validate_items,
            )
            self._writer_created_timestamp = time.time()
        return self.__writer

    def add_priority_table(self, table_name: str, priority_fn: Optional[PriorityFn]):
        if table_name in self._priority_fns:
            raise ValueError(
                f'A priority function already exists for {table_name}. '
                f'Existing tables: {", ".join(self._priority_fns.keys())}.'
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self, timeout_ms: Optional[int] = None):
        """Resets the adder's buffer."""
        if self.__writer:
            # Flush all appended data and clear the buffers.
            self.__writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)

            # Create a new writer unless the current one is too young.
            # This is to reduce the relative overhead of creating a new Reverb writer.
            if (
                time.time() - self._writer_created_timestamp
                > _MIN_WRITER_LIFESPAN_SECONDS
            ):
                self.__writer = None
        self._add_first_called = False

    def add_first(self, timestep: dm_env.TimeStep):
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        self._writer.append(
            dict(observation=timestep.observation, start_of_episode=timestep.first()),
            partial_step=True,
        )
        self._add_first_called = True

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ):
        """Record an action and the following timestep."""

        if not self._add_first_called:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        has_extras = (
            len(extras) > 0 if isinstance(extras, Sized) else extras is not None
        )
        current_step = dict(
            # Observation was passed at the previous add call.
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
            **({"extras": extras} if has_extras else {}),
        )
        self._writer.append(current_step)

        # Record the next observation and write.
        self._writer.append(
            dict(
                observation=next_timestep.observation,
                start_of_episode=next_timestep.first(),
            ),
            partial_step=True,
        )
        self._write()

        if next_timestep.last():
            # Complete the row by appending zeros to remaining open fields.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            self._writer.append(dummy_step)
            self._write_last()
            self.reset()

    @classmethod
    def signature(
        cls, environment_spec: specs.EnvironmentSpec, extras_spec: types.NestedSpec = ()
    ):
        """This is a helper method for generating signatures for Reverb tables.

        Signatures are useful for validating data types and shapes, see Reverb's
        documentation for details on how they are used.

        Args:
          environment_spec: A `specs.EnvironmentSpec` whose fields are nested
            structures with leaf nodes that have `.shape` and `.dtype` attributes.
            This should come from the environment that will be used to generate
            the data inserted into the Reverb table.
          extras_spec: A nested structure with leaf nodes that have `.shape` and
            `.dtype` attributes. The structure (and shapes/dtypes) of this must
            be the same as the `extras` passed into `ReverbAdder.add`.

        Returns:
          A `Step` whose leaf nodes are `tf.TensorSpec` objects.
        """
        spec_step = Step(
            observation=environment_spec.observations,
            action=environment_spec.actions,
            reward=environment_spec.rewards,
            discount=environment_spec.discounts,
            start_of_episode=specs.Array(shape=(), dtype=bool),
            extras=extras_spec,
        )
        return tree.map_structure_with_path(spec_like_to_tensor_spec, spec_step)

    @abc.abstractmethod
    def _write(self):
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self):
        """Write data to replay from the buffer."""


class EndBehavior(enum.Enum):
    """Class to enumerate available options for writing behavior at episode ends.

    Example:

      sequence_length = 3
      period = 2

    Episode steps (digits) and writing events (W):

               1 2 3 4 5 6
                   W   W

    First two sequences:

               1 2 3
               . . 3 4 5

    Written sequences for the different end of episode behaviors:
    Here are the last written sequences for each end of episode behavior:

     WRITE     . . . 4 5 6
     CONTINUE  . . . . 5 6 F
     ZERO_PAD  . . . . 5 6 0
     TRUNCATE  . . . . 5 6

    Key:
      F: First step of the next episode
      0: Zero-filled Step
    """

    WRITE = "write_buffer"
    CONTINUE = "continue_to_next_episode"
    ZERO_PAD = "zero_pad_til_next_write"
    TRUNCATE = "write_truncated_buffer"


class SequenceAdder(ReverbAdder):
    """An adder which adds sequences of fixed length.

    The main differences between this class and the acme sequence adder are:
    (1) for MuZero, we need to pad random actions instead of 0
        after the end of episode.
    (2) added an initial padding option at the beginning of each episode to
    allow easy learner-side frame stacking
    (3) added an end of episode padding option to incorporate mz planning, which
    should continue even after terminal steps are reached
    (4) added functionalities to incorporate the reanalyse episodes:
      (i) at the starting of the episode, pad the history frames as init instead
          of zeros.
      (ii) at the end of the episode, if true ending is reached, pad zeros. if not
          it does not pad anything.
    """

    def __init__(
        self,
        client: reverb.Client,
        sequence_length: int,
        period: int,
        *,
        delta_encoded: bool = False,
        priority_fns: Optional[PriorityFnMapping] = None,
        max_in_flight_items: Optional[int] = 2,
        end_of_episode_behavior: Optional[EndBehavior] = None,
        discrete_action_space: int = 18,
        # Deprecated kwargs.
        chunk_length: Optional[int] = None,
        pad_end_of_episode: Optional[bool] = None,
        break_end_of_episode: Optional[bool] = None,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        extras_spec: Optional[dict] = None,  # pylint: disable=g-bare-generic
        validate_items: bool = True,
        init_padding: int = 0,
        end_padding: int = 1,
    ):
        """Makes a SequenceAdder instance.

        Args:
          client: See docstring for BaseAdder.
          sequence_length: The fixed length of sequences we wish to add.
          period: The period with which we add sequences. If less than
            sequence_length, overlapping sequences are added. If equal to
            sequence_length, sequences are exactly non-overlapping.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          priority_fns: See docstring for BaseAdder.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.
          end_of_episode_behavior:  Determines how sequences at the end of the
            episode are handled (default `EndOfEpisodeBehavior.ZERO_PAD`). See
            the docstring for `EndOfEpisodeBehavior` for more information.
          discrete_action_space: Discrete number of actions in the action space.
          chunk_length: Deprecated and unused.
          pad_end_of_episode: If True (default) then upon end of episode the current
            sequence will be padded (with observations, actions, etc... whose values
            are 0) until its length is `sequence_length`. If False then the last
            sequence in the episode may have length less than `sequence_length`.
          break_end_of_episode: If 'False' (True by default) does not break
            sequences on env reset. In this case 'pad_end_of_episode' is not used.
          environment_spec: environment specs. If we are using frame stacking, then
            put the spec without stacking.
          extras_spec: extra specs
          validate_items: Whether to validate items against the table signature
            before they are sent to the server. This requires table signature to be
            fetched from the server and cached locally.
          init_padding: if we are not using frame stacking, then we need to pad the
            beginning of the env with (num of frames to stack - 1) zeros.
          end_padding: because MZ need to keep predicting after terminal, we pad
            unroll steps number of zeros right after the terminal state. default is
            1 because we always add a dummy step at the end of episode.
        """
        del chunk_length
        super().__init__(
            client=client,
            # We need an additional space in the buffer for the partial step the
            # ReverbAdder will add with the next observation.
            max_sequence_length=sequence_length + 1,
            delta_encoded=delta_encoded,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
            validate_items=validate_items,
        )

        if pad_end_of_episode and not break_end_of_episode:
            raise ValueError(
                "Can't set pad_end_of_episode=True and break_end_of_episode=False at"
                " the same time, since those behaviors are incompatible."
            )

        self._period = period
        self._sequence_length = sequence_length
        self._discrete_action_space = discrete_action_space
        self._environment_spec = environment_spec
        self._extras_spec = extras_spec
        self._init_padding = init_padding
        self._end_padding = end_padding

        if end_of_episode_behavior and (
            pad_end_of_episode is not None or break_end_of_episode is not None
        ):
            raise ValueError(
                "Using end_of_episode_behavior and either "
                "pad_end_of_episode or break_end_of_episode is not permitted. "
                "Please use only end_of_episode_behavior instead."
            )

        # Set pad_end_of_episode and break_end_of_episode to default values.
        if end_of_episode_behavior is None and pad_end_of_episode is None:
            pad_end_of_episode = True
        if end_of_episode_behavior is None and break_end_of_episode is None:
            break_end_of_episode = True

        self._end_of_episode_behavior = EndBehavior.ZERO_PAD
        if pad_end_of_episode is not None or break_end_of_episode is not None:
            if not break_end_of_episode:
                self._end_of_episode_behavior = EndBehavior.CONTINUE
            elif break_end_of_episode and pad_end_of_episode:
                self._end_of_episode_behavior = EndBehavior.ZERO_PAD
            elif break_end_of_episode and not pad_end_of_episode:
                self._end_of_episode_behavior = EndBehavior.TRUNCATE
            else:
                raise ValueError(
                    "Reached an unexpected configuration of the SequenceAdder "
                    f"with break_end_of_episode={break_end_of_episode} "
                    f"and pad_end_of_episode={pad_end_of_episode}."
                )
        elif isinstance(end_of_episode_behavior, EndBehavior):
            self._end_of_episode_behavior = end_of_episode_behavior
        else:
            raise ValueError(
                "end_of_episod_behavior must be an instance of "
                f"EndBehavior, received {end_of_episode_behavior}."
            )

    def reset(self):
        """Resets the adder's buffer."""
        # If we do not write on end of episode, we should not reset the writer.
        if self._end_of_episode_behavior is EndBehavior.CONTINUE:
            return

        super().reset()

    def _write(self, end_of_episode: bool = False):
        self._maybe_create_item(self._sequence_length, end_of_episode=end_of_episode)

    def _write_last(self, full_step):
        # Maybe determine the delta to the next time we would write a sequence.
        # TODO: create zero steps instead of using full step
        if self._end_of_episode_behavior in (
            EndBehavior.TRUNCATE,
            EndBehavior.ZERO_PAD,
        ):
            delta = self._sequence_length - self._writer.episode_steps
            if delta < 0:
                delta = (self._period + delta) % self._period

        # Handle various end-of-episode cases.
        if self._end_of_episode_behavior is EndBehavior.CONTINUE:
            self._maybe_create_item(self._sequence_length, end_of_episode=True)

        elif self._end_of_episode_behavior is EndBehavior.WRITE:
            # Drop episodes that are too short.
            if self._writer.episode_steps < self._sequence_length:
                return
            self._maybe_create_item(
                self._sequence_length, end_of_episode=True, force=True
            )

        elif self._end_of_episode_behavior is EndBehavior.TRUNCATE:
            self._maybe_create_item(
                self._sequence_length - delta, end_of_episode=True, force=True
            )

        elif self._end_of_episode_behavior is EndBehavior.ZERO_PAD:
            for _ in range(delta):
                full_step["action"] = np.random.randint(
                    self._discrete_action_space, dtype=np.int32
                )
                self._writer.append(full_step)

            self._maybe_create_item(
                self._sequence_length, end_of_episode=True, force=True
            )
        else:
            raise ValueError(
                f"Unhandled end of episode behavior: {self._end_of_episode_behavior}."
                " This should never happen, please contact Acme dev team."
            )

    def _maybe_create_item(
        self, sequence_length: int, *, end_of_episode: bool = False, force: bool = False
    ):
        # Check conditions under which a new item is created.
        first_write = self._writer.episode_steps == sequence_length
        # NOTE: the following line assumes that the only way sequence_length
        # is less than self._sequence_length, is if the episode is shorter than
        # self._sequence_length.
        period_reached = self._writer.episode_steps > self._sequence_length and (
            (self._writer.episode_steps - self._sequence_length) % self._period == 0
        )

        if not first_write and not period_reached and not force:
            return

        if not end_of_episode:
            get_traj = operator.itemgetter(slice(-sequence_length - 1, -1))
        else:
            get_traj = operator.itemgetter(slice(-sequence_length, None))

        history = self._writer.history
        trajectory = Trajectory(**tree.map_structure(get_traj, history))

        # Compute priorities for the buffer.
        table_priorities = calculate_priorities(self._priority_fns, trajectory)

        # Create a prioritized item for each table.
        for table_name, priority in table_priorities.items():
            self._writer.create_item(table_name, priority, trajectory)
            self._writer.flush(self._max_in_flight_items)

    @classmethod
    def signature(
        cls,
        environment_spec: specs.EnvironmentSpec,
        extras_spec: types.NestedSpec = (),
        sequence_length: Optional[int] = None,
    ):
        """This is a helper method for generating signatures for Reverb tables.

        Signatures are useful for validating data types and shapes, see Reverb's
        documentation for details on how they are used.

        Args:
          environment_spec: A `specs.EnvironmentSpec` whose fields are nested
            structures with leaf nodes that have `.shape` and `.dtype` attributes.
            This should come from the environment that will be used to generate
            the data inserted into the Reverb table.
          extras_spec: A nested structure with leaf nodes that have `.shape` and
            `.dtype` attributes. The structure (and shapes/dtypes) of this must
            be the same as the `extras` passed into `ReverbAdder.add`.
          sequence_length: An optional integer representing the expected length of
            sequences that will be added to replay.

        Returns:
          A `Trajectory` whose leaf nodes are `tf.TensorSpec` objects.
        """

        def add_time_dim(paths: Iterable[str], spec: tf.TensorSpec):
            return tf.TensorSpec(
                shape=(sequence_length, *spec.shape),
                dtype=spec.dtype,
                name="/".join(str(p) for p in paths),
            )

        trajectory_env_spec, trajectory_extras_spec = tree.map_structure_with_path(
            add_time_dim, (environment_spec, extras_spec)
        )

        spec_step = Trajectory(
            *trajectory_env_spec,
            start_of_episode=tf.TensorSpec(
                shape=(sequence_length,), dtype=tf.bool, name="start_of_episode"
            ),
            extras=trajectory_extras_spec,
        )

        return spec_step

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ):
        """Record an action and the following timestep."""

        if not self._add_first_called:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        has_extras = (
            len(extras) > 0 if isinstance(extras, Sized) else extras is not None
        )
        current_step = dict(
            # Observation was passed at the previous add call.
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
            **({"extras": extras} if has_extras else {}),
        )
        self._writer.append(current_step)

        # Record the next observation and write.
        self._writer.append(
            dict(
                observation=next_timestep.observation,
                start_of_episode=next_timestep.first(),
            ),
            partial_step=True,
        )
        self._write()

        if next_timestep.last():
            # CASE 1: not reanalyse episode, pad unroll steps of zero
            # CASE 2: reanalyse episode but not true end, don't pad
            # CASE 3: reanalyse episode with true end, pad unroll steps of zero
            # Complete the row by appending zeros to remaining open fields.
            full_step = dict(
                # Observation was passed at the previous add call.
                observation=next_timestep.observation,
                start_of_episode=False,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                # Start of episode indicator was passed at the previous add call.
                **({"extras": extras} if has_extras else {}),
            )
            full_step = tree.map_structure(np.zeros_like, full_step)

            dummy_step = tree.map_structure(np.zeros_like, current_step)
            # randomize action
            dummy_action = np.random.randint(
                self._discrete_action_space, dtype=np.int32
            )
            dummy_step["action"] = dummy_action
            self._writer.append(dummy_step)
            self._write(end_of_episode=True)

            append_padding = False

            if next_timestep.last():
                append_padding = True

            if append_padding:
                for _ in range(self._end_padding - 1):
                    full_action = np.random.randint(
                        self._discrete_action_space, dtype=np.int32
                    )
                    full_step["action"] = full_action
                    self._writer.append(full_step)
                    self._write(end_of_episode=True)

            self._write_last(full_step)
            self.reset()

    def add_first(self, timestep: dm_env.TimeStep):
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        if (
            self._environment_spec is not None
            and self._extras_spec is not None
            and self._init_padding > 0
        ):
            zero_observations = jax.tree_map(
                lambda x: jnp.zeros(x.shape, x.dtype), self._environment_spec
            )
            zero_extras = jax.tree_map(
                lambda x: jnp.zeros(x.shape, x.dtype), self._extras_spec
            )
            zero_step = dict(
                # Observation was passed at the previous add call.
                observation=zero_observations.observations,
                start_of_episode=False,
                action=zero_observations.actions,
                reward=zero_observations.rewards,
                discount=zero_observations.discounts,
                # Start of episode indicator was passed at the previous add call.
                **({"extras": zero_extras}),
            )

            for _ in range(self._init_padding):
                self._writer.append(zero_step)

        self._writer.append(
            dict(observation=timestep.observation, start_of_episode=timestep.first()),
            partial_step=True,
        )

        self._add_first_called = True
