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
"""Optimizer utilities."""
from typing import Callable, List, NamedTuple, Optional, Text

import jax
import jax.numpy as jnp
import optax
import config
import flax

import networks

Predicate = Callable[[Text, Text, jnp.ndarray], bool]
INCLUDE_NAMES = ["kernel", "embedding"]
EXCLUDE_NAMES = ["bias"]


def make_optimizer(hparams):
    """Makes the optimizer."""
    if hparams.optimizer == "sgd":
        scale_by = optax.trace(decay=0.9, nesterov=False)
    elif hparams.optimizer == "adam":
        scale_by = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)
    else:
        raise ValueError(f"Unexpected optimizer: {hparams.optimizer}")

    if hparams.clip_value > 0:
        clip_type = hparams.clip_type
        if clip_type == "local":
            clipper = optax.clip(hparams.clip_value)
        elif clip_type == "global_norm":
            clipper = optax.clip_by_global_norm(hparams.clip_value)
        elif clip_type == "adaptive":
            clipper = optax.adaptive_grad_clip(hparams.clip_value)
        elif clip_type == "block_rms":
            clipper = optax.clip_by_block_rms(hparams.clip_value)
        else:
            raise ValueError(f"unknown clip_type {clip_type}")
    else:
        clipper = optax.identity()

    # removing global clipper
    global_clipper = optax.identity()

    weight_decay_scale = hparams.weight_decay_scale / hparams.init_lr
    if hparams.weight_decay_type == "optimizer_update" and weight_decay_scale > 0:
        decayer = add_weight_decay(
            weight_decay_scale,
            include_names=hparams.weight_decay_include_names,
            exclude_names=hparams.weight_decay_exclude_names,
        )
    else:
        decayer = optax.identity()

    if hparams.clip_order == "clip_first":
        return optax.chain(global_clipper, clipper, scale_by, decayer)
    elif hparams.clip_order == "scale_first":
        return optax.chain(global_clipper, scale_by, clipper, decayer)
    else:
        raise ValueError(f"unknown clip order ${hparams.clip_order}")


class AddWeightDecayState(NamedTuple):
    """Stateless transformation."""


def add_weight_decay(
    weight_decay: float,
    include_names: Optional[List[Text]] = None,
    exclude_names: Optional[List[Text]] = None,
) -> optax.GradientTransformation:
    """Adds parameter scaled by `weight_decay` to the `updates`.

    Same as optax.additive_weight_decay but can exclude some parameters.

    Args:
      weight_decay: weight_decay coefficient.
      include_names: an optional list of names to include for weight_decay. ['w']
        by default.
      exclude_names: an optional list of names to exclude for weight_decay. ['b']
        by default.

    Returns:
      An (init_fn, update_fn) tuple.

    """

    def init_fn(_) -> AddWeightDecayState:
        return AddWeightDecayState()

    def update_fn(
        updates,
        state,
        params,
    ):
        exclude = _weight_decay_exclude(
            include_names=include_names, exclude_names=exclude_names
        )
        updates_flat = flax.traverse_util.flatten_dict(updates)
        params_flat = flax.traverse_util.flatten_dict(params)
        u_ex, u_in = _partition(updates_flat, exclude)
        _, p_in = _partition(params_flat, exclude)
        u_in = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, u_in, p_in)
        u_in.update(u_ex)
        updates = flax.traverse_util.unflatten_dict(u_in)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _partition(params_flat, exclude):
    part_exclude = {}
    part_include = {}
    for module_name, value in params_flat.items():
        if exclude(module_name, value):
            part_exclude[module_name] = value
        else:
            part_include[module_name] = value
    return part_exclude, part_include


def _weight_decay_exclude(
    include_names: Optional[List[Text]] = None,
    exclude_names: Optional[List[Text]] = None,
) -> Predicate:
    """Logic for deciding which parameters to include for weight decay..

    Args:
      include_names: an optional list of names to include for weight_decay. ['w']
        by default.
      exclude_names: an optional list of names to exclude for weight_decay. ['b']
        by default.

    Returns:
      A predicate that returns True for params that need to be excluded from
      weight_decay.
    """
    # By default weight_decay the weights but not the biases.
    if not include_names:
        include_names = INCLUDE_NAMES
    if not exclude_names:
        exclude_names = EXCLUDE_NAMES

    # example: ('repr_model', '_res_blocks_9', 'blocks_1', 'layer', 'kernel')
    def exclude(module_name: tuple, value: jnp.array):
        del value
        for m in module_name:
            if networks.NO_WEIGHT_DECAY in m:
                return True

        for m in module_name:
            if m in include_names:
                return False
            elif m in exclude_names:
                return True

        raise ValueError(
            "Parameter in module=%s neither in include_names=%s nor "
            "exclude_names=%s" % (module_name, include_names, exclude_names)
        )

    return exclude


def get_weight_norm(params_flat, include_names, exclude_names) -> jnp.ndarray:
    """Like optimizers.l2_norm, but can exclude some parameters."""
    # include_names: Optional[List[Text]] = None
    exclude = _weight_decay_exclude(
        include_names=include_names, exclude_names=exclude_names
    )

    l2_norm = 0.0
    for module_name, value in params_flat.items():
        if not exclude(module_name, value):
            l2_norm += jnp.sum(jnp.square(value))
    return 0.5 * l2_norm


def get_learning_rate_schedule(
    step: int, config: config.OptimConfig, total_training_steps: int
) -> Optional[float]:
    """Returns the learning rate at step."""
    if config.warmup_steps > 0:
        warmup_multiplier = jnp.minimum(step, config.warmup_steps) / config.warmup_steps
    else:
        warmup_multiplier = 1.0

    init_lr = config.init_lr * warmup_multiplier

    if config.lr_decay_schedule == "exponential":
        return init_lr * config.lr_decay_rate ** (
            jnp.maximum(step - config.lr_decay_after, 0) / float(config.lr_decay_steps)
        )

    # Warning: lr_decay_steps is not used in linear and cosine
    # decay schedule.

    elif config.lr_decay_schedule == "cosine":
        step = jnp.minimum(step, total_training_steps)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step / total_training_steps))
        return init_lr * cosine_decay

    elif config.lr_decay_schedule == "constant":
        return init_lr

    else:
        raise ValueError(
            f"Unexpected learning rate decay schedule: " f"{config.lr_decay_schedule}"
        )
