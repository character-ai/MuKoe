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
from typing import (
    Callable,
    TypeVar,
    Any,
)

from absl import logging
import acme_types as types
import jax
import jax.numpy as jnp
import numpy as np
import ops


F = TypeVar("F", bound=Callable)
N = TypeVar("N", bound=types.NestedArray)
T = TypeVar("T")


NUM_PREFETCH_THREADS = 1


def compute_td_priority(sequence: Any, discount: float = 0.997) -> float:
    """Computes the priority for a sequence based on n-step TD."""
    # TODO: make discount configurable!
    # sequence.discount is 1 if non-terminal, o.w. 0
    discount = sequence.discount * discount
    nonzero_index = np.nonzero(discount)
    discount = discount[nonzero_index]
    reward = sequence.reward[nonzero_index]
    raw_value = sequence.extras["raw_value"]
    raw_value = raw_value[nonzero_index]
    sequence_length = len(discount)
    if sequence_length > 1:
        reward = reward[:-1]
        # [ 1 ] + discount[1:]
        discount = np.insert(discount[1:], 0, 1, axis=0)
        cum_discount = np.cumprod(discount)
        mc_return = (
            np.sum(reward * cum_discount[:-1]) + cum_discount[-1] * raw_value[-1]
        )
    else:
        # If `sequence` only contains a single item then it is the terminal step
        # only. When expanded the "full sequence" would include padding right of the
        # terminal step. However since the padding would set both reward and
        # discount to zero the mc_return will be zero.
        mc_return = 0.0

    # The priority is computed over the raw_value.
    if sequence_length > 1:
        value = raw_value[0]
    else:
        value = 0.0

    # If there a value transformation apply it.
    # TODO: make epsilon configurable
    epsilon = 0.001
    mc_return = ops.value_transformation(mc_return, epsilon)
    value = ops.value_transformation(value, epsilon)

    return abs(value - mc_return)


class MzMctsTemperatureSchedule:
    """Mcts temperature schedule for mz net."""

    def __init__(self, total_training_steps):
        self._total_training_steps = total_training_steps

    def get_temperature(
        self, num_moves: int, training_steps: int, is_training: bool = True
    ) -> float:
        """Gets the sampling temperature. See base class for more details."""

        del num_moves

        if is_training:
            if training_steps < 0.4 * self._total_training_steps:
                temperature = 1.0
            elif training_steps < 0.75 * self._total_training_steps:
                temperature = 0.5
            else:
                temperature = 0.25
        else:
            temperature = 0
        return temperature


def write_metrics(writer, metrics, step, log_period):
    """Writes metrics to tensorboard"""
    with jax.spmd_mode("allow_all"):
        if jax.process_index() == 0:
            for metric_name in metrics:
                writer.add_scalar(metric_name, metrics[metric_name], step)

        full_log = step % log_period == 0

        logging.info(metrics)

        if full_log:
            writer.flush()


def add_batch_dim(values: types.Nest) -> types.NestedArray:
    return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


def add_batch_size(values: types.Nest, bs: int) -> types.NestedArray:
    return jax.tree_map(lambda x: jnp.repeat(x, bs, axis=0), values)


def zeros_like(nest: types.Nest, dtype=None) -> types.NestedArray:
    return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)
