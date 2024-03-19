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
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_manager import (
    CheckpointManagerOptions,
)
import socket
import jax
from jax.experimental import multihost_utils
import portpicker
import logging


def _multislice_distribute_initialize():
    """Calls jax.distribute.initialize() with appropriate multislice arguments."""

    def gen_local_ip():
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    def gen_local_ip_nums():
        return [int(num) for num in gen_local_ip().split(":")[-1].split(".")]

    def get_coordinator_ip():
        local_ip_nums = jax.numpy.array(gen_local_ip_nums())
        coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
        coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
        return ".".join(coordinator_ip_strings)

    port = multihost_utils.broadcast_one_to_all(
        jax.numpy.array(portpicker.pick_unused_port())
    )
    coordinator_address = get_coordinator_ip() + ":" + str(port)
    try:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=jax.process_count(),
            process_id=jax.process_index(),
        )
    except RuntimeError:
        logging.info("Jax distributed already initialized")
        pass


def get_ckpt_manager(path, save_interval_steps, create=True, use_async=False):
    # p = epath.Path(path)
    if use_async:
        _multislice_distribute_initialize()
        enable_async_checkpointing = True
    else:
        enable_async_checkpointing = False
    options = CheckpointManagerOptions(
        create=create,
        max_to_keep=10,
        save_interval_steps=save_interval_steps,
        enable_async_checkpointing=enable_async_checkpointing,
    )
    mngr = ocp.CheckpointManager(
        path, options=options, item_handlers=ocp.StandardCheckpointHandler()
    )
    logging.info("ckpt manager created!")
    return mngr
