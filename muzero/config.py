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
"""IMPALA config."""
import dataclasses
from typing import Tuple


@dataclasses.dataclass
class ModelConfig:
    """model configuration for MZ."""

    output_channels: int = 256
    num_layers: int = 10
    input_resolution: Tuple[int, int] = (96, 96)
    target_resolution: Tuple[int, int] = (6, 6)
    action_space: int = 18
    dynamics_num_layers: int = 6


@dataclasses.dataclass
class TrainConfig:
    """training configuration for MZ."""

    td_steps: int = 10
    num_unroll_steps: int = 5
    batchsize: int = 512
    total_training_steps: int = 1_000_000
    log_period: int = 10
    ckpt_save_interval_steps: int = 50
    # TEST:
    # batchsize: int = 8
    # log_period: int = 1
    # total_training_steps: int = 100
    # ckpt_save_interval_steps: int = 4


@dataclasses.dataclass
class ReplayConfig:
    """training configuration for MZ."""

    replay_sequence_length: int = 160
    max_replay_size: int = 100000
    min_fill_fraction: float = 0.01
    samples_per_insert: int = 8
    # TEST:
    # replay_sequence_length: int = 120
    # max_replay_size: int = 320000
    # min_fill_fraction: int = 0.0001
    # samples_per_insert: int = 1


@dataclasses.dataclass
class InferenceConfig:
    """training configuration for MZ."""

    dyna_batch_size: int = 256
    repr_batch_size: int = 8
    dyna_time_out: float = 0.002
    repr_time_out: float = 0.001
    dyna_update_interval: int = (
        16000000  # repr_update_interval * num_simulations in mcts/utils.py
    )
    repr_update_interval: int = 1500
    dyna_actor_per_replica: int = 60
    repr_actor_per_replica: int = 60


@dataclasses.dataclass
class OptimConfig:
    lr_decay_rate: float = 0.3
    lr_decay_steps: int = int(300e3)
    lr_decay_after: int = 100_000
    value_loss_type: str = "ce"
    init_lr: float = 1e-4
    optimizer: str = "adam"
    lr_decay_schedule: str = "cosine"
    # weight_decay_scale: float = 1e-4
    weight_decay_scale: float = 0.0
    weight_decay_type: str = "loss_penalty"
    weight_decay_include_names: dict = dataclasses.field(
        default_factory=lambda: ["kernel", "embedding"]
    )
    weight_decay_exclude_names: dict = dataclasses.field(
        default_factory=lambda: ["bias"]
    )
    clip_value: float = 1.0
    agc: float = 0.0
    clip_type: str = "local"
    clip_order: str = "scale_first"
    clip_by_global_norm: float = 32.0
    warmup_steps: int = 1000


@dataclasses.dataclass
class LossConfig:
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    value_loss_type: str = "ce"
    reward_loss_weight: float = 1.0
    reward_loss_type: str = "ce"
    use_raw_value: bool = True


@dataclasses.dataclass
class ParallelismConfig:
    dcn_data_parallelism: int = -1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    # ICI stands for "Intercore interconnect" the interconnect within a slice (fast)
    ici_data_parallelism: int = 8
    ici_fsdp_parallelism: int = 1
    ici_tensor_parallelism: int = 1
