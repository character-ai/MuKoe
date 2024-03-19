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
"""Stateless operations on JAX or numpy arrays."""
import dataclasses

import jax
import jax.numpy as jnp
import rlax

MIN_VALUE_TRANSFORMATION_EPS = 1e-8


@dataclasses.dataclass
class ValueTransformationOptions:
    min_value: float
    max_value: float
    num_bins: int
    value_transformation_epsilon: float


def value_transformation_options(
    value_transformation_epsilon: float = 0.001,
) -> ValueTransformationOptions:
    """Returns the value transformation options."""
    return ValueTransformationOptions(
        max_value=300.0,
        min_value=-300.0,
        num_bins=601,
        value_transformation_epsilon=value_transformation_epsilon,
    )


def inverse_value_transformation(x, eps: float):
    """Implements the inverse of the R2D2 value transformation."""
    _check_value_transformation_eps(eps)
    return rlax.signed_parabolic(x, eps) if eps != 0.0 else x


def _check_value_transformation_eps(eps: float) -> None:
    """Throws if the epsilon for value transformation isn't valid."""
    if eps < 0.0:
        raise ValueError("-ve epsilon ({}) not supported".format(eps))
    elif 0 < eps < MIN_VALUE_TRANSFORMATION_EPS:
        raise ValueError(
            "0 < eps < {} not supported ({})".format(MIN_VALUE_TRANSFORMATION_EPS, eps)
        )


def value_transformation(x, eps: float):
    """Implements the R2D2 value transformation."""
    _check_value_transformation_eps(eps)
    return rlax.signed_hyperbolic(x, eps) if eps != 0 else x


def clip_gradient(x: jnp.ndarray, abs_value: float) -> jnp.ndarray:
    """Clips the gradient of `x` to be in [-abs_value, abs_value]."""

    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda g: (jnp.clip(g, -abs_value, abs_value),)

    return wrapped(x)
