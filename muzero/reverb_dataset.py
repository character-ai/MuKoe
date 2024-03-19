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
import collections
import os
from typing import Callable, Mapping, Optional, Union, NamedTuple, Any

import adder
import reverb
import tensorflow as tf
from typing import Iterator
import atari_env_loop

import acme_types as types
import numpy as np
import tree
import time
import networks
import logging
import jax
from jax._src.partition_spec import PartitionSpec
from jax._src.mesh import Mesh

Pytree = Any
Device = Any
Transform = Callable[[reverb.ReplaySample], reverb.ReplaySample]


class ReverbData(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    discount: Any
    extras: Any


def make_reverb_dataset(
    server_address: str,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    table: Union[str, Mapping[str, float]] = adder.DEFAULT_PRIORITY_TABLE,
    num_parallel_calls: Optional[int] = 12,
    max_in_flight_samples_per_worker: Optional[int] = None,
    postprocess: Optional[Transform] = None,
) -> tf.data.Dataset:
    """Make a TensorFlow dataset backed by a Reverb trajectory replay service.

    Arguments:
      server_address: Address of the Reverb server.
      batch_size: Batch size of the returned dataset.
      prefetch_size: The number of elements to prefetch from the original dataset.
        Note that Reverb may do some internal prefetching in addition to this.
      table: The name of the Reverb table to use, or a mapping of (table_name,
        float_weight) for mixing multiple tables in the input (e.g. mixing online
        and offline experiences).
      num_parallel_calls: The parralelism to use. Setting it to `tf.data.AUTOTUNE`
        will allow `tf.data` to automatically find a reasonable value.
      max_in_flight_samples_per_worker: see reverb.TrajectoryDataset for details.
      postprocess: User-specified transformation to be applied to the dataset (as
        `ds.map(postprocess)`).

    Returns:
      A `tf.data.Dataset` iterating over the contents of the Reverb table.

    Raises:
      `table` is a
      mapping with no positive weight values.
    """

    # This is the default that used to be set by reverb.TFClient.dataset().
    if max_in_flight_samples_per_worker is None and batch_size is None:
        max_in_flight_samples_per_worker = 100
    elif max_in_flight_samples_per_worker is None:
        max_in_flight_samples_per_worker = 2 * batch_size

    # Create mapping from tables to non-zero weights.
    if isinstance(table, str):
        tables = collections.OrderedDict([(table, 1.0)])
    else:
        tables = collections.OrderedDict(
            [(name, weight) for name, weight in table.items() if weight > 0.0]
        )
        if len(tables) <= 0:
            raise ValueError(f"No positive weights in input tables {tables}")

    # Normalize weights.
    total_weight = sum(tables.values())
    tables = collections.OrderedDict(
        [(name, weight / total_weight) for name, weight in tables.items()]
    )

    def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
        datasets = ()
        for table_name, weight in tables.items():
            max_in_flight_samples = max(
                1, int(max_in_flight_samples_per_worker * weight)
            )
            dataset = reverb.TrajectoryDataset.from_table_signature(
                server_address=server_address,
                table=table_name,
                max_in_flight_samples_per_worker=max_in_flight_samples,
            )
            datasets += (dataset,)
        if len(datasets) > 1:
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets, weights=tables.values()
            )
        else:
            dataset = datasets[0]

        # Post-process each element if a post-processing function is passed, e.g.
        # observation-stacking or data augmenting transformations.
        if postprocess:
            dataset = dataset.map(postprocess)

        if batch_size:
            dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    if num_parallel_calls is not None:
        # Create a datasets and interleaves it to create `num_parallel_calls`
        # `TrajectoryDataset`s.
        num_datasets_to_interleave = (
            os.cpu_count()
            if num_parallel_calls == tf.data.AUTOTUNE
            else num_parallel_calls
        )
        dataset = tf.data.Dataset.range(num_datasets_to_interleave).interleave(
            map_func=_make_dataset,
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
    else:
        dataset = _make_dataset(tf.constant(0))

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


class NumpyIterator(Iterator[types.NestedArray]):
    """Iterator over a dataset with elements converted to numpy.

    Note: This iterator returns read-only numpy arrays.

    This iterator (compared to `tf.data.Dataset.as_numpy_iterator()`) does not
    copy the data when comverting `tf.Tensor`s to `np.ndarray`s.

    """

    __slots__ = ["_iterator"]

    def __init__(self, dataset):
        self._iterator: Iterator[types.NestedTensor] = iter(dataset)

    def __iter__(self) -> "NumpyIterator":
        return self

    def __next__(self) -> types.NestedArray:
        return tree.map_structure(
            lambda t: np.asarray(memoryview(t)), next(self._iterator)
        )

    def next(self):
        return self.__next__()


def temp_compute_value(sequence, extras, discount=0.997):
    # TODO: make discount = 0.997 configurable
    reward = sequence.reward
    raw_value = []
    for i in range(len(reward)):
        # go from the back
        if i == 0:
            current_value = extras[networks.RAW_VALUE][-1 - i]
        else:
            current_value = reward[-1 - i] + discount * current_value
        raw_value.insert(0, current_value)
    return raw_value


def _get_mz_inputs(
    sample: reverb.ReplaySample,
    sequence_length: int,
    stack_frame: int = 1,
    use_raw_value: bool = False,
    discount: float = 0.997,
) -> reverb.ReplaySample:
    """Returns MzFeatures."""
    # stack_frame: if set to 1, then do not stack frame, that is, assume env side
    #   already stacks frames.

    data = sample.data
    extras = data.extras

    if use_raw_value:
        raw_value = temp_compute_value(data, extras, discount)
        extras[networks.RAW_VALUE] = tf.convert_to_tensor(raw_value)

    terminal_idx = tf.math.reduce_sum(tf.cast(sample.data.discount > 0, tf.int32)) + 1
    # upper bound is either sequence_length before the end of the long sequence
    # or the terminal state
    upper_bound = tf.math.minimum(
        data.discount.shape[0] - sequence_length + 1 - stack_frame + 1, terminal_idx
    )
    idx_seed = (time.time_ns()) % 1e9
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=upper_bound, dtype=tf.int32, seed=idx_seed
    )
    sequence_start_idx = idx + stack_frame - 1
    sequence_slice = slice(sequence_start_idx, sequence_start_idx + sequence_length)

    new_extras = {}
    new_extras[networks.POLICY_PROBS] = extras[networks.POLICY_PROBS][sequence_slice]
    new_extras[networks.NETWORK_STEPS] = extras[networks.NETWORK_STEPS][sequence_slice]
    new_extras[networks.RAW_VALUE] = extras[networks.RAW_VALUE][sequence_slice]

    subsampled_new_observation = data.observation[
        idx : sequence_start_idx + sequence_length
    ]
    if stack_frame > 1:
        subsampled_new_observation = tf.concat(
            [
                tf.roll(subsampled_new_observation, -1 * i, axis=0)
                for i in range(stack_frame)
            ],
            axis=-1,
        )
        subsampled_new_observation = subsampled_new_observation[: -(stack_frame - 1)]
    new_reverb_data = ReverbData(
        observation=subsampled_new_observation,
        action=data.action[sequence_slice],
        reward=data.reward[sequence_slice],
        discount=data.discount[sequence_slice],
        extras=new_extras,
    )
    return reverb.ReplaySample(info=sample.info, data=new_reverb_data)


def make_mz_dataset(
    dataset: tf.data.Dataset,
    batch_size: int,
    prefetch_size: int,
    sequence_length: int,
    num_parallel_calls: int = 16,
    use_raw_value: bool = False,
    discount: float = 0.997,
) -> tf.data.Dataset:
    """Returns a dataset of MzFeatures."""
    logging.info("processing")
    dataset = dataset.map(
        lambda x: _get_mz_inputs(
            x,
            sequence_length,
            atari_env_loop.ATARI_NUMBER_STACK_FRAME,
            use_raw_value,
            discount,
        ),
        num_parallel_calls,
    )
    # TODO: add end of episode padding and reduce sequence length to
    # TD-step length.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.prefetch(prefetch_size)
    return dataset


def get_next_batch_sharded(
    local_data,
    data_sharding,
    global_data_shape: Pytree,
    global_mesh: Mesh,
) -> jax.Array:
    """Splits the host loaded data equally over all devices."""

    # local_devices = jax.local_devices()
    local_devices = global_mesh.local_devices
    local_device_count = jax.local_device_count()

    def _put_to_devices(x):
        try:
            per_device_arrays = np.split(x, local_device_count, axis=0)
        except ValueError as array_split_error:
            raise ValueError(
                f"Unable to put to devices shape {x.shape} with "
                f"local device count {local_device_count}"
            ) from array_split_error
        device_buffers = [
            jax.device_put(arr, d) for arr, d in zip(per_device_arrays, local_devices)
        ]
        return device_buffers

    # 'fully shard' the data (first) axis across both axes
    # of the hardware mesh. This is layout matches the
    # manual device placing we just did.
    input_sharding_constraint = PartitionSpec(*data_sharding, None)

    def form_gda(local_data, shape):
        device_buffers = _put_to_devices(local_data)
        #  Wrap device buffers as GDA
        shape = tuple(shape)
        input_gda = jax.make_array_from_single_device_arrays(
            shape,
            jax.sharding.NamedSharding(global_mesh, input_sharding_constraint),
            device_buffers,
        )
        return input_gda

    input_gdas = jax.tree_map(form_gda, local_data, global_data_shape)

    return input_gdas
