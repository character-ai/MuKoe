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
"""Base class for MCTS."""

import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union, Generic, TypeVar

import numpy as np


@dataclasses.dataclass(frozen=True)
class PredictionFnOutput:
    """The output of prediction network."""

    # The predicted value.
    value: Union[float, int]
    # The predicted per step reward.
    reward: Union[float, int]
    # The predicted action logits.
    action_logits: Optional[np.ndarray] = None


EmbeddingT = TypeVar("EmbeddingT")


@dataclasses.dataclass(frozen=True)
class ModelFunctions(Generic[EmbeddingT]):
    """A collection of functions that are used by the search."""

    # Callable function for the representation net. Given an observation, it
    # returns the embeddings of the observation and the prediction.
    repr_and_pred: Callable[[Any], Tuple[EmbeddingT, PredictionFnOutput]]

    # Callable function for dynamics net and prediction net, it returns the next
    # state and predictions given the current state and action.
    dyna_and_pred: Callable[[Any, EmbeddingT], Tuple[EmbeddingT, PredictionFnOutput]]

    # Callable function for returning a list of legal action masks, 1 for valid
    # actions, 0 for invalid actions. Input is the action history stored as a list
    # of numbers.
    get_legal_actions_mask: Callable[[List[Union[float, int]]], np.ndarray]
