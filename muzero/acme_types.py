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
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
import specs

NestedArray = Any
NestedTensor = Any

# The pmap axis name. Data means data parallelization.
PMAP_AXIS_NAME = "data"

NestedSpec = Union[
    specs.Array,
    Iterable["NestedSpec"],
    Mapping[Any, "NestedSpec"],
]

Nest = Union[NestedArray, NestedTensor, NestedSpec]

TensorTransformation = Callable[[NestedTensor], NestedTensor]
TensorValuedCallable = Callable[..., NestedTensor]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()
