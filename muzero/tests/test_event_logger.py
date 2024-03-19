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
"""Tests for event_logger.py.

Run this with:
python3 -m pytest -v -rP tests/test_event_logger.py
"""

import pytest
import ray
import event_logger
import time
import sys


def test_basic_usage():
    ray.init()
    event_logger.initialize()
    owner_name = "tester"
    event_category = "testCase"
    event_id = "1"
    args = dict(owner_name=owner_name, event_category=event_category, event_id=event_id)
    event_logger.event_start(**args)
    time.sleep(2)
    event_logger.event_stop(**args)
    event_logger.summary()
    time.sleep(2)
    ray.shutdown()


def test_multiple_calls():
    ray.init()
    event_logger.initialize()
    owner_name = "tester"
    event_category = "testCase"
    args = dict(owner_name=owner_name, event_category=event_category)
    for _ in range(10):
        event_logger.event_start(**args)
        time.sleep(0.5)
        event_logger.event_stop(**args)

    event_logger.summary()
    time.sleep(3)
    ray.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
