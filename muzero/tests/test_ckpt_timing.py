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
import unittest
import time
import orbax.checkpoint as ocp


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        # out of date:
        # ckpt_dir = "gs://yejingxin-us-central2-public/wendy/ckpt2"
        ckpt_dir = "gs://character-ai-us-central1/ckpt_wendy/seaquest_0826_test_new/"
        starttime = time.time()
        print(f"ckpt dir {ckpt_dir}")
        options = ocp.CheckpointManagerOptions()
        _ckpt_manager = ocp.CheckpointManager(
            ckpt_dir,
            item_handlers=ocp.StandardCheckpointHandler(),
            options=options,
        )
        all_steps = _ckpt_manager.all_steps(read=True)
        print(f"find all ckpts spent {time.time() - starttime}s")
        starttime = time.time()
        latest_step = max(all_steps) if all_steps else None
        if latest_step is None:
            latest_step = 0
        print("all steps")
        print(all_steps)
        restored = _ckpt_manager.restore(latest_step)
        restored_params = restored["save_state"]["state"]
        print(restored_params)
        print(f"restore ckpt takes spent {time.time() - starttime}s")


if __name__ == "__main__":
    unittest.main()
