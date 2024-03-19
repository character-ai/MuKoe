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
"""Utils class for MCTS."""

from dataclasses import field, dataclass


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for normalization based on running statistics.

    Attributes:
      max_abs: Maximum value for clipping.
      statistics_update_period: How often to update running statistics used for
        normalization.
    """

    current_phase: int = 0
    num_simulations: int = 50
    #### Test
    # num_simulations: int = 5
    action_space_size: int = 18
    # Use parent's value as q when the node's visit count is zero. It is used
    # to compute the node's UCB score during child node selection.
    parent_base_visit_count: int = 1
    use_parent_value_as_q: bool = False
    use_parent_value_as_q_steps: int = 1_000_000
    # Virtual loss. Setting it to 0 to disable it.
    virtual_loss: int = 10
    # Noise added to the node value. Setting it to 0 to disable it.
    # value_noise: float=0.5
    value_noise: float = 0.0
    # Prior temperature
    prior_temperature: float = 1.5
    # Number of simulations shares the same virtual loss.
    num_simulation_share_virtual_loss: int = 4
    # mcts temperature (after search).
    disable_value_normalization: bool = False
    root_exploration_blur: float = 0.0
    use_softmax_for_action_selection: bool = False
    dirichlet: dict = field(
        default_factory=lambda: {
            "alpha": 0.03,
            "exploration_fraction": 0.25,
        }
    )
    ucb: dict = field(
        default_factory=lambda: {
            "pb_c_init": 1.25,
            "pb_c_base": 19652,
        }
    )
    value_head: dict = field(
        default_factory=lambda: {
            "discount": 0.99,
        }
    )


def get_default_mcts_params() -> MCTSConfig:
    """Returns default hyper-parameters for mcts."""
    # MCTS search described in https://arxiv.org/abs/1911.08265 Appendix B
    return MCTSConfig()


class MzMctsTemperatureSchedule:
    """Mcts temperature schedule for mz net."""

    def __init__(self, total_training_steps):
        self._total_training_steps = total_training_steps

    def get_temperature(self, training_steps: int, is_training: bool = True) -> float:
        """Gets the sampling temperature."""

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
