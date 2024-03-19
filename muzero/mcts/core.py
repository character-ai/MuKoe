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
# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension
"""MuZero Core.

Based partially on https://arxiv.org/src/1911.08265v1/anc/pseudocode.py

"""

import collections
from typing import List, MutableMapping, Optional, Set, Tuple

from absl import flags
import numpy as np
import scipy.ndimage
from mcts.utils import MCTSConfig


FLAGS = flags.FLAGS
MAXIMUM_FLOAT_VALUE = float("inf")

NetworkOutput = collections.namedtuple(
    "NetworkOutput",
    "value value_logits reward reward_logits policy_logits hidden_state",
)

Prediction = collections.namedtuple(
    "Prediction", "gradient_scale value value_logits reward reward_logits policy_logits"
)

Target = collections.namedtuple(
    "Target", "value_mask reward_mask policy_mask value reward visits"
)

Range = collections.namedtuple("Range", "low high")

Action = np.int64  # pylint: disable=invalid-name


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, root_value=None, disabled=False):
        self.maximum = -MAXIMUM_FLOAT_VALUE
        self.minimum = MAXIMUM_FLOAT_VALUE
        self.disabled = disabled
        if root_value is not None:
            self.maximum = root_value
            self.minimum = root_value

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.disabled:
            return value
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            value = (value - self.minimum) / (self.maximum - self.minimum)
        value = max(min(1.0, value), 0.0)
        return value


class TransitionModel:
    """Transition model providing additional information for MCTS transitions.

    An environment can provide a specialized version of a transition model via the
    info dict. This model then provides additional information, e.g. on the legal
    actions, between transitions in the MCTS.
    """

    def __init__(self, full_action_space_size: int):
        self.full_action_space_size = full_action_space_size

    def legal_actions_after_sequence(self, actions_sequence: Optional[Tuple[int]]):  # pylint: disable=unused-argument
        """Returns the legal action space after a sequence of actions."""
        return range(self.full_action_space_size)

    def full_action_space(self):
        return range(self.full_action_space_size)

    def legal_actions_mask_after_sequence(self, actions_sequence: Optional[Tuple[int]]):
        """Returns the legal action space after a sequence of actions as a mask."""
        mask = np.zeros(self.full_action_space_size, dtype=np.int64)
        for action in self.legal_actions_after_sequence(actions_sequence):
            mask[action] = 1
        return mask


class Node:
    """Node for MCTS."""

    def __init__(
        self,
        config: MCTSConfig,
        is_root=False,
        child_priors: Optional[np.ndarray] = None,
    ) -> None:
        self.visit_count = 0
        self.is_root = is_root
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.discount = config.value_head["discount"]
        self.child_priors = child_priors
        self.virtual_loss = config.virtual_loss

    def expanded(self) -> bool:
        return self.child_priors is not None

    def get_visit_count(self, with_virtual_loss: bool = False) -> float:
        """Gets the node visit count."""
        if with_virtual_loss:
            return self.visit_count + self.virtual_loss
        else:
            return self.visit_count

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def qvalue(self) -> float:
        return self.discount * self.value() + self.reward


class ActionHistory:
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(Action(action))

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]


def prepare_root_node(
    config: MCTSConfig,
    legal_actions: np.ndarray,
    initial_inference_output: NetworkOutput,
) -> Node:
    root = Node(config, is_root=True)
    expand_node(root, legal_actions, initial_inference_output, config.prior_temperature)
    add_exploration_noise(config, root)
    return root


def reset_virtual_loss(nodes: Set[Node]) -> None:
    """Resets the virtual loss of the nodes."""
    for node in nodes:
        node.virtual_loss = 0


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: MCTSConfig,
    root: Node,
    action_history: ActionHistory,
    legal_actions_fn,
    recurrent_inference_fn,
    visualization_fn=None,
) -> MutableMapping[str, float]:
    min_max_stats = MinMaxStats(
        root_value=None, disabled=config.disable_value_normalization
    )
    min_max_stats.update(root.qvalue())

    search_heights = []
    virtual_losses_nodes = set()
    for i in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        if (
            config.virtual_loss > 0
            and i % config.num_simulation_share_virtual_loss == 0
        ):
            reset_virtual_loss(nodes=virtual_losses_nodes)
            virtual_losses_nodes = set()

        # This loop will go a maximum of i iterations (or i - 1)
        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(
                action
            )  # pytype: disable=wrong-arg-types  # numpy-scalars
            search_path.append(node)
            if config.virtual_loss > 0:
                node.virtual_loss += config.virtual_loss
                virtual_losses_nodes.add(node)

        search_heights.append(len(search_path))
        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = recurrent_inference_fn(
            parent.hidden_state, history.last_action()
        )
        # Validate the policy_logits is in the correct shape.
        assert network_output.policy_logits.shape == (config.action_space_size,)
        legal_actions = legal_actions_fn(history.history[len(action_history.history) :])

        # Expand node
        expand_node(
            node, legal_actions, network_output, config.prior_temperature, min_max_stats
        )

        backpropagate(
            search_path,
            network_output.value,
            config.value_head["discount"],
            min_max_stats,
        )

    if config.virtual_loss > 0:
        # Reset virtual loss again for the last few simulations.
        reset_virtual_loss(nodes=virtual_losses_nodes)

    if visualization_fn:
        visualization_fn(root)

    metrics = {}
    metrics["mcts_search_height_mean"] = np.mean(search_heights)
    metrics["mcts_search_height_max"] = np.max(search_heights)
    return metrics


def masked_distribution(
    x: List[float], use_exp: bool, mask: Optional[np.ndarray] = None
):
    if mask is None:
        mask = np.ones(shape=len(x))
    assert sum(mask) > 0, "Not all values can be masked."
    assert len(mask) == len(x), "The dimensions of the mask and x need to be the same."
    x = np.exp(x) if use_exp else np.array(x, dtype=np.float64)
    x *= mask
    if sum(x) == 0:
        # No unmasked value has any weight. Use uniform distribution over unmasked
        # tokens.
        x = mask
    return x / np.sum(x, keepdims=True)


def masked_softmax(x, mask: Optional[np.ndarray] = None, temperature=1.0):
    x = np.array(x) - np.max(x, axis=-1)  # to avoid overflow
    x = x / temperature
    return masked_distribution(x, use_exp=True, mask=mask)


def masked_count_distribution(x, mask: Optional[np.ndarray] = None):
    return masked_distribution(
        x, use_exp=False, mask=mask
    )  # pytype: disable=wrong-arg-types  # always-use-return-annotations


def histogram_sample(
    distribution: List[Tuple[int, int]],
    temperature: float,
    use_softmax: bool = False,
    mask: Optional[np.ndarray] = None,
):
    actions = [d[1] for d in distribution]
    visit_counts = np.array([d[0] for d in distribution], dtype=np.float64)
    if temperature == 0.0:
        probs = masked_count_distribution(visit_counts, mask=mask)
        return actions[np.argmax(probs)]
    if use_softmax:
        logits = visit_counts / temperature
        probs = masked_softmax(logits, mask)
    else:
        logits = visit_counts ** (1.0 / temperature)
        probs = masked_count_distribution(logits, mask)
    return np.random.choice(actions, p=probs)


def select_action(
    node: Node, visit_softmax_temperature: float, use_softmax: bool = False
) -> np.ndarray:
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    return histogram_sample(
        distribution=visit_counts,
        temperature=visit_softmax_temperature,
        use_softmax=use_softmax,
    )


def select_child(
    config: MCTSConfig, node: Node, min_max_stats: MinMaxStats
) -> Tuple[int, Node]:
    """Selects a child action with the highest UCB score.

    Args:
      config: MCTSConfig with MCTS configurations.
      node: The node whose children are being selected.
      min_max_stats: Normalize the q value based on its min & max setting.

    Returns:
      A tuple of selected action and child node.
    """

    child_visit_counts = np.zeros(shape=config.action_space_size)
    if config.use_parent_value_as_q:
        parent_value = min_max_stats.normalize(node.value())
        child_values = np.ones(config.action_space_size) * np.squeeze(parent_value)
    else:
        child_values = np.zeros(shape=config.action_space_size)

    for action, child in node.children.items():
        # validate the expanded child has been visited before.
        assert child.visit_count > 0
        child_visit_counts[action] = child.get_visit_count(with_virtual_loss=True)

        value_score = child.qvalue()
        if config.value_noise > 0:
            value_score += np.random.uniform(
                low=-config.value_noise, high=config.value_noise
            )
        value_score = min_max_stats.normalize(value_score)
        child_values[action] = value_score

    # TODO: try adding children's virtual loss to 
    # the parent node's visit count 
    pb_c = (
        np.log(
            (node.visit_count + config.ucb["pb_c_base"] + 1) / config.ucb["pb_c_base"]
        )
        + config.ucb["pb_c_init"]
    )
    pb_c *= np.sqrt(node.visit_count + config.parent_base_visit_count) / (
        np.asarray(child_visit_counts) + 1
    )

    if node.child_priors is None:
        raise ValueError(
            "child_priors should not be None because MCTS should only selects a"
            "child from an expanded node"
        )

    prior_score = pb_c * node.child_priors
    score = prior_score + child_values

    action = np.argmax(score)
    if action not in node.children:
        node.children[action] = Node(config)
    return action, node.children[action]


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
    node: Node,
    legal_actions: np.ndarray,
    network_output: NetworkOutput,
    prior_temperature: float,
    min_max_stats=None,
):
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy_probs = masked_softmax(
        network_output.policy_logits, mask=legal_actions, temperature=prior_temperature
    )
    node.child_priors = policy_probs
    node.value_sum += network_output.value
    node.visit_count += 1
    if min_max_stats is not None:
        # TODO: make the code cleaner by moving the condition to prepare_root_node
        min_max_stats.update(node.qvalue())


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats
):
    # The leaf node's value has already been updated during expanding node op.
    leaf_node = search_path[-1]
    value = leaf_node.reward + discount * value
    for node in search_path[-2::-1]:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.qvalue())
        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MCTSConfig, node: Node):
    if config.root_exploration_blur > 0:
        node.child_priors = scipy.ndimage.gaussian_filter1d(
            node.child_priors,
            config.root_exploration_blur,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
    node.child_priors /= np.sum(node.child_priors)
    noise = np.random.dirichlet([config.dirichlet["alpha"]] * len(node.child_priors))
    frac = config.dirichlet["exploration_fraction"]
    node.child_priors = node.child_priors * (1 - frac) + noise * frac
