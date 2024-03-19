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
"""MCTS using L2G MCTS library."""

from typing import Any, Tuple, MutableMapping

import numpy as np

from mcts.base import ModelFunctions
from mcts import core
from mcts.utils import MCTSConfig, MzMctsTemperatureSchedule


class L2GSearch:
    """Base search class."""

    def __init__(
        self, params: MCTSConfig, model_fns: ModelFunctions, total_training_steps: int
    ) -> None:
        self._mz_config = params
        self._mcts_temperature_schedule = MzMctsTemperatureSchedule(
            total_training_steps
        )
        self._model_fns = model_fns

    @property
    def mz_config(self):
        return self._mz_config

    def _get_normalized_visit_counts(self, root: core.Node) -> np.ndarray:
        """Gets the normalized visit counts of the child nodes.

        Args:
          root: The root node whose children are used to get the visit counts.

        Returns:
          The normalized visit counts of the child nodes.
        """
        visit_counts = [0.0] * self._mz_config.action_space_size
        for child_action, child in root.children.items():
            visit_counts[child_action] = float(child.visit_count)
        return np.asarray(visit_counts, dtype=np.float32) / np.sum(visit_counts)

    def _get_normalized_value_logits(self, root: core.Node) -> np.ndarray:
        """Gets the normalized values of the child nodes.

        Args:
          root: The root node whose children are used to get the value.

        Returns:
          The normalized values of the child nodes.
        """
        value_bins = [0.0] * self._mz_config.action_space_size
        for child_action, child in root.children.items():
            value_bins[child_action] = float(child.value_sum)

        norm_value = value_bins / np.sum(np.abs(value_bins))
        return np.asarray(norm_value, dtype=np.float32)

    def gen_action(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
        self, observation: Any, training_steps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, MutableMapping[str, float]]:
        """Generate an action. See base class for more details."""
        root, metrics = self._search(observation=observation)

        # The normalized visit counts are used as policy label during training.
        policy_training_logits = self._get_normalized_visit_counts(root=root)

        visit_softmax_temperature = self._mcts_temperature_schedule.get_temperature(
            training_steps
        )

        action = core.select_action(
            node=root,
            visit_softmax_temperature=visit_softmax_temperature,
            use_softmax=self._mz_config.use_softmax_for_action_selection,
        )

        metrics["mcts_sampling_temperature"] = visit_softmax_temperature

        return action.astype(np.int32), policy_training_logits, metrics

    def _search(self, observation: Any) -> Tuple[core.Node, MutableMapping[str, float]]:
        """Create a tree and perform MCTS.

        Args:
          observation: The observation used to call model_fns.

        Returns:
          A tuple of:
          The tree's root node after the search with search stats populated.
          The metrics associated with the tree search.
        """
        root_actions_mask = self._model_fns.get_legal_actions_mask([])

        def _recurrent_inference_fn(embedding, action):
            dyna_out, pred_out = self._model_fns.dyna_and_pred(embedding, action)

            # Value and reward logits are set to None because they are not used.
            return core.NetworkOutput(
                value=pred_out.value,
                value_logits=None,
                reward=pred_out.reward,
                reward_logits=None,
                policy_logits=pred_out.action_logits,
                hidden_state=dyna_out,
            )

        init_repr_out, init_pred_out = self._model_fns.repr_and_pred(observation)
        # Setting the initial reward to 0 because the reward of the trajectory is
        # not available at the root node. Value and reward logits are set to none
        # because they are not used.
        init_out = core.NetworkOutput(
            value=init_pred_out.value,
            value_logits=None,
            reward=0,
            reward_logits=None,
            policy_logits=init_pred_out.action_logits,
            hidden_state=init_repr_out,
        )

        action_history = core.ActionHistory(
            history=[], action_space_size=self._mz_config.action_space_size
        )

        root = core.prepare_root_node(self._mz_config, root_actions_mask, init_out)

        metrics = core.run_mcts(
            config=self._mz_config,
            root=root,
            action_history=action_history,
            legal_actions_fn=self._model_fns.get_legal_actions_mask,
            recurrent_inference_fn=_recurrent_inference_fn,
        )

        # the raw value is the initial estimated value of the root, from the value
        # network. It will be used to calculate priority
        metrics["raw_value"] = init_pred_out.value
        return root, metrics
