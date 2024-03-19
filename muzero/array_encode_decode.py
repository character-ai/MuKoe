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
import msgpack
import numpy as np
import jax


def _dtype_from_name(name: str):
    """Handle JAX bfloat16 dtype correctly."""
    if name == b"bfloat16":
        return jax.numpy.bfloat16
    else:
        return np.dtype(name)


def ndarray_to_bytes(arr) -> bytes:
    """Save ndarray to simple msgpack encoding."""
    if isinstance(arr, jax.Array):
        arr = np.array(arr)
    if arr.dtype.hasobject or arr.dtype.isalignedstruct:
        raise ValueError(
            "Object and structured dtypes not supported "
            "for serialization of ndarrays."
        )
    tpl = (arr.shape, arr.dtype.name, arr.tobytes("C"))
    return msgpack.packb(tpl, use_bin_type=True)


def ndarray_from_bytes(data: bytes) -> np.ndarray:
    """Load ndarray from simple msgpack encoding."""
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    return np.frombuffer(
        buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0
    ).reshape(shape, order="C")
