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
import dm_env
import numpy as np
import acme_types as types
import tree
import specs
from typing import Optional, Sequence


def _generate_from_spec(spec: types.NestedSpec) -> types.NestedArray:
    """Generate a value from a potentially nested spec."""
    return tree.map_structure(lambda s: _normalize_array(s).generate_value(), spec)


def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
    """Validate a value from a potentially nested spec."""
    tree.assert_same_structure(value, spec)
    tree.map_structure(lambda s, v: s.validate(v), spec, value)


def _normalize_array(array: specs.Array) -> specs.Array:
    """Converts bounded arrays with (-inf,+inf) bounds to unbounded arrays.

    The returned array should be mostly equivalent to the input, except that
    `generate_value()` returns -infs on arrays bounded to (-inf,+inf) and zeros
    on unbounded arrays.

    Args:
      array: the array to be normalized.

    Returns:
      normalized array.
    """
    if isinstance(array, specs.DiscreteArray):
        return array
    if not isinstance(array, specs.BoundedArray):
        return array
    if not (array.minimum == float("-inf")).all():
        return array
    if not (array.maximum == float("+inf")).all():
        return array
    return specs.Array(array.shape, array.dtype, array.name)


class Environment(dm_env.Environment):
    """A fake environment with a given spec."""

    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        *,
        episode_length: int = 25,
    ):
        # Assert that the discount spec is a BoundedArray with range [0, 1].
        def check_discount_spec(path, discount_spec):
            if (
                not isinstance(discount_spec, specs.BoundedArray)
                or not np.isclose(discount_spec.minimum, 0)
                or not np.isclose(discount_spec.maximum, 1)
            ):
                if path:
                    path_str = " " + "/".join(str(p) for p in path)
                else:
                    path_str = ""
                raise ValueError(
                    "discount_spec {}isn't a BoundedArray in [0, 1].".format(path_str)
                )

        tree.map_structure_with_path(check_discount_spec, spec.discounts)

        self._spec = spec
        self._episode_length = episode_length
        self._step = 0

    def _generate_fake_observation(self):
        return _generate_from_spec(self._spec.observations)

    def _generate_fake_reward(self):
        return _generate_from_spec(self._spec.rewards)

    def _generate_fake_discount(self):
        return _generate_from_spec(self._spec.discounts)

    def reset(self) -> dm_env.TimeStep:
        observation = self._generate_fake_observation()
        self._step = 1
        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        _validate_spec(self._spec.actions, action)

        observation = self._generate_fake_observation()
        reward = self._generate_fake_reward()
        discount = self._generate_fake_discount()

        if self._episode_length and (self._step == self._episode_length):
            self._step = 0
            # We can't use dm_env.termination directly because then the discount
            # wouldn't necessarily conform to the spec (if eg. we want float32).
            return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, observation)
        else:
            self._step += 1
            return dm_env.transition(
                reward=reward, observation=observation, discount=discount
            )

    def action_spec(self):
        return self._spec.actions

    def observation_spec(self):
        return self._spec.observations

    def reward_spec(self):
        return self._spec.rewards

    def discount_spec(self):
        return self._spec.discounts


class _BaseDiscreteEnvironment(Environment):
    """Discrete action fake environment."""

    def __init__(
        self,
        *,
        num_actions: int = 1,
        action_dtype=np.int32,
        observation_spec: types.NestedSpec,
        discount_spec: Optional[types.NestedSpec] = None,
        reward_spec: Optional[types.NestedSpec] = None,
        **kwargs,
    ):
        """Initialize the environment."""
        if reward_spec is None:
            reward_spec = specs.Array((), np.float32)

        if discount_spec is None:
            discount_spec = specs.BoundedArray((), np.float32, 0.0, 1.0)

        actions = specs.DiscreteArray(num_actions, dtype=action_dtype)

        super().__init__(
            spec=specs.EnvironmentSpec(
                observations=observation_spec,
                actions=actions,
                rewards=reward_spec,
                discounts=discount_spec,
            ),
            **kwargs,
        )


class DiscreteEnvironment(_BaseDiscreteEnvironment):
    """Discrete state and action fake environment."""

    def __init__(
        self,
        *,
        num_actions: int = 1,
        num_observations: int = 1,
        action_dtype=np.int32,
        obs_dtype=np.int32,
        obs_shape: Sequence[int] = (),
        discount_spec: Optional[types.NestedSpec] = None,
        reward_spec: Optional[types.NestedSpec] = None,
        **kwargs,
    ):
        """Initialize the environment."""
        observations_spec = specs.BoundedArray(
            shape=obs_shape,
            dtype=obs_dtype,
            minimum=obs_dtype(0),
            maximum=obs_dtype(num_observations - 1),
        )

        super().__init__(
            num_actions=num_actions,
            action_dtype=action_dtype,
            observation_spec=observations_spec,
            discount_spec=discount_spec,
            reward_spec=reward_spec,
            **kwargs,
        )
