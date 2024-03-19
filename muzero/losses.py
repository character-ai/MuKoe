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
import abc
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import optax
import rlax
import config
import flax

import networks
from flax import core
import ops
import optimizers
import reverb_dataset

POLICY_LOSS = "policy_loss"
VALUE_LOSS = "value_loss"
REWARD_LOSS = "reward_loss"
REGULARIZATION_LOSS = "regularization_loss"
TOTAL_LOSS = "total_loss"

REWARDS = "rewards"


class DistributionSupport(abc.ABC):
    """Object that describes the support of a discrete distribution."""

    def __init__(self, min_value: float, max_value: float, num_bins: int):
        self._min_value = min_value
        self._max_value = max_value
        self._num_bins = num_bins

    def min_value(self) -> float:
        return self._min_value

    def max_value(self) -> float:
        return self._max_value

    def num_bins(self) -> int:
        return self._num_bins

    @abc.abstractmethod
    def bin_to_value(self, bin_value: jnp.ndarray) -> jnp.ndarray:
        """Returns the value associated with a bin."""
        pass

    @abc.abstractmethod
    def scalar_to_two_hot(self, value: jnp.ndarray) -> jnp.ndarray:
        """Returns a distribution repr of this value for this support."""
        pass

    def scalar_to_gaussian(self, value: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
        raise ValueError("Scalar to Gaussian not implemented.")

    def mean(self, distribution: jnp.ndarray) -> jnp.ndarray:
        """Returns the mean of this distribution for this support."""
        assert len(distribution.shape) == 1
        bin_indices = jnp.arange(self._num_bins)
        bin_values = self.bin_to_value(bin_indices)
        return jnp.vdot(bin_values, distribution.astype(jnp.float32))

    def _truncate_value(self, value: jnp.ndarray) -> jnp.ndarray:
        """Truncates values outside the  min max range."""
        return jnp.maximum(self._min_value, jnp.minimum(self._max_value, value))

    def _scalar_to_two_hot(self, target_value: jnp.ndarray, target_bin: jnp.ndarray):
        """Builds a distribution repr for target_value/bin for this support.

        The distribution is built by assigning probabilities on adjacent
        bins to target_bin so that the target_value can be recovered excatly by
        target_value = l_value * (1-p_upper) + u_value * p_upper.

        Args:
          target_value: value to be transformed to a distribution
          target_bin: associated bin to this value

        Returns:
          A distribution representing this value based on this support.
        """
        assert target_value.shape == target_bin.shape
        assert len(target_value.shape) == 0  # pylint: disable=g-explicit-length-test

        eps = 1e-10
        l_bin, u_bin = jnp.floor(target_bin), jnp.ceil(target_bin)
        l_value, u_value = self.bin_to_value(l_bin), self.bin_to_value(u_bin)

        p_upper = jnp.where(
            l_bin == u_bin, 0.5, (target_value - l_value) / (u_value - l_value + eps)
        )

        l_dist = jax.nn.one_hot(l_bin, self._num_bins)
        u_dist = jax.nn.one_hot(u_bin, self._num_bins)
        return (1 - p_upper) * l_dist + p_upper * u_dist


class LinearDistributionSupport(DistributionSupport):
    """Support spread linearly, e.g. [-2, -1, 0, 1, 2]."""

    def __init__(self, min_value: float, max_value: float, num_bins: int):
        super().__init__(min_value, max_value, num_bins)
        self._bin_size = (self._max_value - self._min_value) / (self._num_bins - 1.0)

    def bin_size(self) -> float:
        return self._bin_size

    def bin_to_value(self, bin_value: jnp.ndarray) -> jnp.ndarray:
        return self._min_value + bin_value * self._bin_size

    def scalar_to_two_hot(self, value: jnp.ndarray) -> jnp.ndarray:
        target_value = self._truncate_value(value)
        target_bin = (target_value - self._min_value) / self._bin_size
        return self._scalar_to_two_hot(target_value, target_bin)

    def scalar_to_gaussian(self, value: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
        def normal(x):
            stddev = scale * self._bin_size
            return 0.5 * jax.lax.erf((x - value) / (jnp.sqrt(2) * stddev))

        bins = self.bin_to_value(jnp.arange(self._num_bins))

        width = 0.5 * self._bin_size
        probs = normal(bins + width) - normal(bins - width)
        probs /= jnp.sum(probs)

        two_hot = self.scalar_to_two_hot(value)
        return jnp.where(
            value < self._max_value,
            jnp.where(value > self._min_value, probs, two_hot),
            two_hot,
        )


def softmax_cross_entropy_loss(labels: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """Computes the softmax cross entropy loss.

    Args:
      labels: Labels in [B, Size of the action space].
      logits: Logits in [B, Size of the action space].

    Returns:
      The loss in []
    """
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.mean(loss)


def categorical_loss(
    labels: jnp.ndarray, logits: jnp.ndarray, support: DistributionSupport
):
    """Categorical cross entropy loss with some scalar evaluations."""
    # Build the categorical label
    label_dist_fn = support.scalar_to_two_hot
    label_dist = jax.vmap(label_dist_fn)(labels)

    # Compute the cross entropy loss
    ce = softmax_cross_entropy_loss(label_dist, logits)

    # Store a few more evaluation summaries
    avg = jax.vmap(support.mean)(jax.nn.softmax(logits))
    mse_scalar = mean_squared_error(avg, labels)

    # Return the loss and the evaluation summaries
    return ce, mse_scalar


def mean_squared_error(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Computes the mean squared error.

    Args:
      predictions: Predictions in [B, ]
      labels: Labels in [B, ]

    Returns:
      The error in []
    """
    return jnp.mean(jnp.square(predictions - labels))


def quantile_huber_loss(
    output_samples: jnp.ndarray,
    output_taus: jnp.ndarray,
    target_samples: jnp.ndarray,
    delta: float = 1.0,
) -> jnp.ndarray:
    """Quantile huber loss, smoother version of the regular quantile loss.

    Args:
      output_samples: samples from our estimate of the target distribution.
      output_taus: the quantiles associated to the output_samples.
      target_samples: samples from the target distribution.
      delta: smoothing parameter controlling the huber loss.

    Returns:
      A loss for every sample in every batch.
    """
    diff = target_samples - output_samples
    huber_factor = optax.huber_loss(diff, delta=delta)

    indicator = (diff < 0.0).astype(diff.dtype)
    indicator = jax.lax.stop_gradient(indicator)
    quantile_factor = jnp.abs(output_taus - indicator)
    return quantile_factor * huber_factor


def get_loss_and_metrics(
    network: networks.MzNet,
    loss_params: config.LossConfig,
    unroll_step: int,
    discount: float,
    data: reverb_dataset.ReverbData,
    params: core.FrozenDict[str, Any],
    target_params: core.FrozenDict[str, Any],
    weight_decay_type: str = "",
    weight_decay_scale: float = 0.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Returns the loss and metrics on the entire episode.

    Args:
      network: a BaseMzNetwork implementation.
      sample: Reverb sample data.
      params: params of the networks, these are pure params not states
      target_params: params of the target networks, these are pure params not states
      loss_params: configurations for losses.
      optim_params: configurations for optimization.
      unroll_step: number of unroll steps
      discount: td discount.
      rng: Random number generator key.

    Returns:
      a tuple of loss and metrics_dict

    Raises:
      ValueError: An error occurred when the number of actions in the feature does
        not equal to the number of unroll steps.
    """

    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )
    # policy prob does not need to be masked, for we have extra['policy_probs']
    # to be 0's after terminal state
    policy_probs = extra[networks.POLICY_PROBS]
    mask = jnp.where(discounts > 0, 1.0, 0.0)
    discounts = mask * discount
    batch_size = policy_probs.shape[0]
    step_index = -1

    # the last ation should not be used
    actions = actions[:, :unroll_step]

    predictions, predictions_state = network.apply(
        {"params": params},
        observations[:, 0],
        actions,
        jnp.float32,
        mutable=["activations"],
    )

    if loss_params.use_raw_value:
        values_final = extra[networks.RAW_VALUE][:, -1]
        value_targets = extra[networks.RAW_VALUE]
    else:
        # construct value target
        obs_final = observations[jnp.arange(batch_size), step_index]
        obs_final_output = network.apply(
            {"params": target_params}, obs_final, method=network.repr_and_pred
        )
        values_final = obs_final_output[1]["value"]
        values_final = jax.lax.stop_gradient(values_final)
        value_targets = jax.vmap(rlax.discounted_returns)(
            rewards, discounts, values_final
        )

    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    reward_loss = 0.0

    metrics_dict = {}
    metrics_dict["value_final"] = jnp.mean(values_final)
    metrics_dict["debug_value_final"] = (
        extra[networks.NETWORK_STEPS],
        discounts,
        policy_probs[:, 1, :],
        predictions[1][networks.POLICY],
        value_targets,
        predictions[1]["value"],
        rewards,
        predictions[1]["reward"],
    )
    for t, pred_t in enumerate(predictions):
        # policy loss
        policy_logits_t = pred_t[networks.POLICY]  # [B, num_actions]
        policy_labels_t = policy_probs[:, t, :]
        policy_loss_t = softmax_cross_entropy_loss(
            labels=policy_labels_t, logits=policy_logits_t
        )  # [B]
        policy_loss += policy_loss_t
        metrics_dict[POLICY_LOSS + "_" + str(t)] = policy_loss_t

        # reward loss for t > 0, = 0 is not necessary, but im hoping to help with training
        if t >= 0:
            if loss_params.reward_loss_type == "ce":
                reward_label_t = rewards[:, t]
                reward_label_transform_t = ops.value_transformation(
                    reward_label_t, 0.001
                )
                reward_loss_t, reward_mse_t = categorical_loss(
                    labels=reward_label_transform_t,
                    logits=pred_t["reward_logits"],
                    support=LinearDistributionSupport(
                        min_value=-300.0,
                        max_value=300.0,
                        num_bins=601,
                    ),
                )
                reward_loss += reward_loss_t
                metrics_dict[REWARD_LOSS + "_mse_" + str(t)] = reward_mse_t
                metrics_dict[REWARD_LOSS + "_" + str(t)] = reward_loss_t
                metrics_dict["reward_label_" + str(t)] = jnp.mean(reward_label_t)
                metrics_dict["reward_pred_" + str(t)] = jnp.mean(pred_t["reward"])

        # value loss
        if loss_params.value_loss_type == "ce":
            value_label_t = value_targets[:, t]
            value_label_transfrom_t = ops.value_transformation(value_label_t, 0.001)
            value_loss_t, value_mse_t = categorical_loss(
                labels=value_label_transfrom_t,
                logits=pred_t["value_logits"],
                support=LinearDistributionSupport(
                    min_value=-300.0,
                    max_value=300.0,
                    num_bins=601,
                ),
            )
            value_loss += value_loss_t  # []
            metrics_dict[VALUE_LOSS + "_mse_" + str(t)] = value_mse_t
            metrics_dict[VALUE_LOSS + "_" + str(t)] = value_loss_t
            metrics_dict["value_label" + "_" + str(t)] = jnp.mean(value_label_t)
            metrics_dict["value_pred" + "_" + str(t)] = jnp.mean(pred_t["value"])
        else:
            raise ValueError(
                f"Unexpected value type: " f"{loss_params.value_loss_type}"
            )

    total_loss += policy_loss * loss_params.policy_loss_weight
    total_loss += value_loss * loss_params.value_loss_weight
    total_loss += reward_loss * loss_params.reward_loss_weight
    metrics_dict[POLICY_LOSS] = policy_loss
    metrics_dict[VALUE_LOSS] = value_loss
    metrics_dict[REWARD_LOSS] = reward_loss

    if weight_decay_type == "loss_penalty":
        regularization = 0.0
        params_flat = flax.traverse_util.flatten_dict(params)
        regularization += optimizers.get_weight_norm(
            params_flat,
            include_names=optimizers.INCLUDE_NAMES,
            exclude_names=optimizers.EXCLUDE_NAMES,
        )
        total_loss += regularization * weight_decay_scale
        metrics_dict[REGULARIZATION_LOSS] = regularization

    metrics_dict[REWARDS] = jnp.mean(rewards)
    metrics_dict[TOTAL_LOSS] = total_loss

    return total_loss, metrics_dict
