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
from typing import (
    Any,
    Callable,
    List,
    MutableMapping,
    Tuple,
)

import acme_types as types

import jax.numpy as jnp
import flax.linen as nn
import flax
import jax
import math
import functools
import numpy as np
import ops
import rlax
import enum
import config
import jax_types
from dm_env import specs
import utils

# Constants for prediction heads.
POLICY = "policy"
VALUE = "value"
REWARD = "reward"

# Constants for the extra values in the replay buffer.
POLICY_PROBS = "policy_probs"
NETWORK_STEPS = "network_steps"
# TODO: use a different name from step type overloaded notation
RAW_VALUE = "raw_value"
REANALYSE_SAMPLE = 0
ONLINE_SAMPLE = 1
_TARGET_RESOLUTION = 10

NO_WEIGHT_DECAY = "no_weight_decay"

Params = MutableMapping[str, Any]


class EmbedLookupStyle(enum.Enum):
    """How to return the embedding matrices given IDs."""

    ARRAY_INDEX = 1
    ONE_HOT = 2


NonLinearity = Callable[[jnp.ndarray], jnp.ndarray]


def _to_scalar(
    output: jax.numpy.array, trans_options: ops.ValueTransformationOptions
) -> jnp.ndarray:
    """Produces the scalar representation of the output."""
    output = jax.nn.softmax(output)
    scalar_output = rlax.transform_from_2hot(
        output, trans_options.min_value, trans_options.max_value, trans_options.num_bins
    )
    if trans_options.value_transformation_epsilon > 0:
        scalar_output = ops.inverse_value_transformation(
            scalar_output, trans_options.value_transformation_epsilon
        )
    return scalar_output


class ResBlock(nn.Module):
    r"""Creates a residual block."""

    output_channels: int
    stack_size: int
    stride: int = 1
    non_linearity: NonLinearity = jax.nn.relu

    def setup(self):
        self._last_non_linearity = lambda x: x

        blocks = []
        for i in range(self.stack_size):
            blocks.append(ConvBlock(self.output_channels, with_bias=False))
        self.blocks = blocks

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        output = x
        for block in self.blocks:
            output = block(output)
        output = self._last_non_linearity(x + output)
        return output


class ConvBlock(nn.Module):
    r"""Creates a basic conv block."""

    output_channels: int
    stride: int = 1
    is_first: bool = False
    with_bias: bool = True
    non_linearity: NonLinearity = jax.nn.relu

    def setup(self):
        self.layer = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=self.stride,
            kernel_init=flax.linen.initializers.normal(0.01),
            use_bias=self.with_bias,
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        output = x
        if not self.is_first:
            output = flax.linen.LayerNorm(
                epsilon=1e-6, use_scale=True, use_bias=True, name=f"{NO_WEIGHT_DECAY}"
            )(output)
            output = self.non_linearity(output)

        output = self.layer(output)
        return output


def make_basic_conv_blocks(
    conv_planes: List[int], non_linearity: NonLinearity = jax.nn.relu
):
    """Creates a list of basic convolutional blocks.

    Args:
      conv_planes: A list of output planes for each conv block.
      order: Order to pass to a BasicBlock.
      make_norm_op: Normalization op.
      **conv_kwargs: Optional kwargs for `make_conv`, such as `w_init`.

    Returns:
      A list of Blocks.
    """
    convs = []
    for _, p in enumerate(conv_planes):
        convs.append(ConvBlock(p, non_linearity=non_linearity))
    return convs


def make_downsample_layers(
    output_channels: int,
    resolution: Tuple[int, int],
    target_resolution: int = _TARGET_RESOLUTION,
):
    """Creates layers to downsample input images to <= 6x6."""

    def get_channels(resolution: Tuple[int, int]):
        if max(resolution) > 48:
            # At high resolutions we downsample at a reduced channel count.
            return min(128, output_channels)
        else:
            # Otherwise we downsample at the full channel count.
            return output_channels

    # Input is usually 96x96.
    cur_channels = -1
    layers = []
    is_first = True

    while max(resolution) > target_resolution:
        channels = get_channels(resolution)
        stride = (2, 2)
        new_res = tuple(int(math.ceil(r / s)) for r, s in zip(resolution, stride))

        if max(resolution) >= 48 or cur_channels != output_channels:
            # At high resolutions or when we need to change the number of channels
            # we downsample with a conv.
            layers.append(ConvBlock(channels, stride=stride, is_first=is_first))
            is_first = False
            cur_channels = channels
        else:
            # Otherwise we downsample by pooling.
            layers.append(
                functools.partial(
                    flax.linen.avg_pool,
                    window_shape=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                )
            )

        # After each downsampling we apply some residual conv blocks.
        for i in range(3):
            layers.append(ResBlock(channels, stack_size=2))

        resolution = new_res

    # Final hidden state would be 6x6 when starting from 96x96.
    return layers


class BaseHead(nn.Module):
    """Basic MuZero head: some 1x1 convolutions followed by linear layers."""

    make_conv_blocks: Callable[..., List[nn.Module]]
    linear_sizes: List[int]

    def setup(self):
        self._conv_blocks = self.make_conv_blocks()
        linears = []
        for i, size in enumerate(self.linear_sizes):
            linears.append(jax.nn.relu)
            linears.append(
                nn.Dense(size, kernel_init=flax.linen.initializers.normal(0.01))
            )
        self._linears = linears

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        for layer in self._conv_blocks:
            x = layer(x)

        x = x.reshape((x.shape[0], -1))

        for layer in self._linears:
            x = layer(x)
        return x


class ConvCategoricalHead(BaseHead):
    """A head that represents continuous values by a categorical distribution."""

    transformation: ops.ValueTransformationOptions

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        logits = super().__call__(x)
        return dict(logits=logits, mean=_to_scalar(logits, self.transformation))


class RepresentationNet(nn.Module):
    """Representation network base class."""

    output_channels: int
    num_layers: int
    input_resolution: Tuple[int, int]
    expected_downsampled: Tuple[int, int]

    def setup(self):
        self._expected_downsampled = self.expected_downsampled
        downsample_channels = self.output_channels

        # Downsample input until it's <= 10x10.
        self._downsampling = make_downsample_layers(
            downsample_channels, self.input_resolution
        )

        res_blocks = []
        for i in range(self.num_layers):
            res_blocks.append(ResBlock(self.output_channels, stack_size=2))
        self._res_blocks = res_blocks

    @nn.compact
    def __call__(self, obs: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        # Start by downsampling the input image.
        obs = jnp.asarray(obs / 255, dtype=dtype)
        obs_shape = obs.shape
        image = jnp.reshape(obs, list(obs_shape[:-2]) + [obs_shape[-1] * obs_shape[-2]])
        for module in self._downsampling:
            image = module(image)
        _, height, width, _ = image.shape  # Update with downsampled dimensions.
        assert (
            (height, width) == self._expected_downsampled
        ), f"downsampled to {(height, width)}; expected {self._expected_downsampled}"
        # Finally, apply the normal residual tower.
        for block in self._res_blocks:
            image = block(image)
        return image


class PredictionNet(nn.Module):
    """A network that makes predictions based on the current hidden state."""

    discrete_classes: int

    def setup(self):
        make_categorical_head = functools.partial(
            ConvCategoricalHead,
            make_conv_blocks=functools.partial(
                make_basic_conv_blocks, conv_planes=[16]
            ),
        )

        policy_planes = [16]
        self._policy_head = BaseHead(
            functools.partial(make_basic_conv_blocks, conv_planes=policy_planes),
            linear_sizes=[256, self.discrete_classes],
        )

        value_transformation = ops.value_transformation_options()

        self._value_head = make_categorical_head(
            linear_sizes=[256, value_transformation.num_bins],
            transformation=value_transformation,
        )

        self._reward_head = make_categorical_head(
            linear_sizes=[256, value_transformation.num_bins],
            transformation=value_transformation,
        )

    @nn.compact
    def __call__(self, embedding: jnp.ndarray):
        outputs = {}
        value = self._value_head(embedding)
        outputs.update(value=value["mean"], value_logits=value["logits"])

        reward = self._reward_head(embedding)
        outputs.update(reward=reward["mean"], reward_logits=reward["logits"])

        policy = self._policy_head(embedding)
        outputs.update(policy=policy)
        return outputs


class ConvDynamicsNet(nn.Module):
    """A network mapping a hidden state + action to the next hidden state."""

    output_channels: int
    num_layers: int
    discrete_classes: int

    def setup(self):
        self._embed_action = Embed(
            vocab_size=self.discrete_classes, embed_dim=self.output_channels
        )
        res_blocks = []
        for i in range(self.num_layers):
            res_blocks.append(ResBlock(self.output_channels, stack_size=2))
        self._res_blocks = res_blocks

    @nn.compact
    def __call__(self, embedding: jnp.ndarray, action: jnp.ndarray):
        _, h, w, _ = embedding.shape
        # Embed the action so that it has as many channels as the embedding.
        action = action.astype(jnp.float32)
        action = self._embed_action(action)
        action = flax.linen.LayerNorm(
            epsilon=1e-3,
            use_scale=True,
            use_bias=True,
            name=f"act_embed_{NO_WEIGHT_DECAY}",
        )(action)
        action = jax.nn.relu(action)
        # Tile the action to match the embedding, then add it residual style.
        action = jnp.tile(action[:, None, None, :], [1, h, w, 1])
        x = embedding + action
        for block in self._res_blocks:
            x = block(x)
        return x


class Embed(nn.Module):
    """An Embedding layer."""

    vocab_size: int
    embed_dim: int
    init_scale: float = 0.0001
    lookup_style: EmbedLookupStyle = EmbedLookupStyle.ARRAY_INDEX

    def setup(self):
        self.embeddings = self.param(
            "embedding",
            nn.initializers.normal(stddev=np.sqrt(self.init_scale / self.embed_dim)),
            (self.vocab_size, self.embed_dim),
            jnp.float32,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, dtype: jnp.dtype = jnp.float32):
        if x.dtype != jnp.int32:
            assert x.dtype == jnp.float32
            x = jnp.round(x).astype(jnp.int32)
        # Workaround for lookup during export for inference.
        embeddings = self.embeddings * jnp.ones_like(self.embeddings)
        if self.lookup_style == EmbedLookupStyle.ARRAY_INDEX:
            return embeddings[(x,)].astype(dtype)
        elif self.lookup_style == EmbedLookupStyle.ONE_HOT:
            one_hot = jax.nn.one_hot(x, self.vocab_size, dtype=dtype)
            return jnp.einsum("...v,vc->...c", one_hot, embeddings)


class MzNet(nn.Module):
    # model_config: config.ModelConfig
    output_channels: int
    num_layers: int
    input_resolution: Tuple[int, int]
    target_resolution: Tuple[int, int]
    action_space: int
    dynamics_num_layers: int

    def setup(self):
        # set up representation net
        self.repr_model = RepresentationNet(
            output_channels=self.output_channels,
            num_layers=self.num_layers,
            input_resolution=self.input_resolution,
            expected_downsampled=self.target_resolution,
        )

        # set up prediction net
        self.pred_model = PredictionNet(self.action_space)
        self.dyna_pred_model = PredictionNet(self.action_space)

        # set up dynamics net
        self.dyna_model = ConvDynamicsNet(
            output_channels=self.output_channels,
            num_layers=self.dynamics_num_layers,
            discrete_classes=self.action_space,
        )

    def dyna_and_pred(self, embedding: jnp.ndarray, action: jnp.ndarray):
        """See details in the base class."""
        e_out = self.dyna_model(embedding=embedding, action=action)
        e_out = jax.tree_map(lambda x: ops.clip_gradient(x, 1.0), e_out)
        prediction = self.dyna_pred_model(embedding=e_out)
        # self.sow("activations", "action_embed2", jax.lax.stop_gradient(e_out))
        return e_out, prediction

    def repr_and_pred(self, obs: types.NestedArray, dtype: jnp.dtype = jnp.float32):
        """See details in the base class."""
        embedding = self.repr_model(obs, dtype)
        prediction = self.pred_model(embedding=embedding)
        return embedding, prediction

    def __call__(
        self,
        obs: types.NestedArray,
        actions: jnp.ndarray,
        dtype: jnp.dtype = jnp.float32,
    ) -> List[types.NestedArray]:
        """See details in the base class."""
        predictions = []
        repr_embedding, repr_prediction = self.repr_and_pred(obs, dtype)
        predictions.append(repr_prediction)
        embedding = repr_embedding
        for i in range(actions.shape[1]):  #  [B, T,]
            dyna_e_out, dyna_prediction = self.dyna_and_pred(
                #  [B,]
                embedding=embedding,
                action=actions[:, i],
            )
            predictions.append(dyna_prediction)
            embedding = dyna_e_out
        return predictions


def init_mz_network(
    network: MzNet, observation_spec: specs.Array, rng: jax_types.PRNGKey
):
    """Initialize the mz network with a dummy observation and action.

    Args:
      network: An implementation of the BaseMzNetwork to initialize.
      observation_spec: The observation spec used to make dummy input.
      rng: The jax random number generator key in the shape of [2].
      call_args: Call arguments.

    Returns:
      The params and states of the network.
    """
    dummy_obs = utils.add_batch_dim(utils.zeros_like(observation_spec))
    dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
    return network.init(rng, obs=dummy_obs.observations, actions=dummy_action)


def get_model(model_config: config.ModelConfig):
    return MzNet(
        output_channels=model_config.output_channels,
        num_layers=model_config.num_layers,
        input_resolution=model_config.input_resolution,
        target_resolution=model_config.target_resolution,
        action_space=model_config.action_space,
        dynamics_num_layers=model_config.dynamics_num_layers,
    )
