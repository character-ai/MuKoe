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
"""A custom logger that tracks timestamps of various events."""
import ray
import time
import numpy as np
import logging
import dataclasses
from collections import defaultdict
from typing import Mapping, Optional


_ENABLE_EVENT_LOGGER = True


@dataclasses.dataclass
class Event:
    start: float
    stop: Optional[float] = None


History = Mapping[
    str, Mapping[str, Event]
]  # owner_name -> event_category -> event_id -> Event


@ray.remote(num_cpus=1, resources={"event_logger_handler": 1})
class EventLogger:
    def __init__(self):
        self.history: History = defaultdict(lambda: defaultdict(dict))

    def event_start(
        self,
        owner_name: str,
        event_category: str,
        timestamp: float,
        event_id: Optional[str] = None,
    ):
        """Records the start of an event.

        Args:
            owner_name: Name of the actor.
            event_category: Category of the event.
            timestamp: The start time of the event.
            event_id: Optional unique identifier for the event. If None,
                an ID will be generated.
        """
        if event_id is None:
            event_id = self.generate_event_id(owner_name, event_category)
        self.history[owner_name][event_category][event_id] = Event(start=timestamp)

    def event_stop(
        self,
        owner_name: str,
        event_category: str,
        timestamp: float,
        event_id: Optional[str] = None,
    ):
        """Records the end of an event.

        Args:
            owner_name: Name of the actor.
            event_category: Category of the event.
            timestamp: The start time of the event.
            event_id: Optional unique identifier for the event. If None,
                an ID will be generated.
        """
        if event_id is None:
            event_id = self.get_latest_event_id(owner_name, event_category)
        if event_id and event_id in self.history[owner_name][event_category]:
            self.history[owner_name][event_category][event_id].stop = timestamp

    def generate_event_id(self, owner_name: str, event_category: str) -> str:
        """
        Generates a unique event ID based on the actor name, event category, and the number of existing events.

        Args:
            owner_name: Name of the actor.
            event_category: Category of the event.

        Returns:
            Generated unique event ID.
        """
        return f"{owner_name}_{event_category}_{len(self.history[owner_name][event_category])}"

    def get_latest_event_id(
        self, owner_name: str, event_category: str
    ) -> Optional[str]:
        """
        Retrieves the latest event ID for the specified actor and category.

        Args:
            owner_name: Name of the actor.
            event_category: Category of the event.

        Returns:
            The latest event ID or None if no events exist for the actor-category combination.
        """
        if self.history[owner_name][event_category]:
            return list(self.history[owner_name][event_category].keys())[-1]
        return None

    def summary(self):
        for actor, actor_history in self.history.items():
            print(f"Actor::{actor}")
            for event_category, event_history in actor_history.items():
                # Extract durations, ignoring events that are not yet completed
                event_durations = [
                    event.stop - event.start
                    for event_id, event in event_history.items()
                    if event.stop is not None
                ]
                # Check if there are any completed events to avoid errors in statistical calculations
                if event_durations:
                    print(f"EventCategory::{event_category}")
                    print(f"AvgTime::{np.mean(event_durations)}")
                    print(f"MedianTime::{np.median(event_durations)}")  # p50
                    print(f"P90::{np.percentile(event_durations, 90)}")
                    print(f"P99::{np.percentile(event_durations, 99)}")
                    print(f"Num invocations: {len(event_durations)}")
                else:
                    print(f"EventCategory::{event_category} - No completed events")
            print("------------------")


def _get_global_logger() -> ray.actor.ActorHandle:
    try:
        global_logger = ray.get_actor("event_logger", namespace="metrics")
    except ValueError as e:
        logging.error("Failed to get global actor: ", e)
        raise e
    return global_logger


def event_start(owner_name: str, event_category: str, event_id: Optional[int] = None):
    if not _ENABLE_EVENT_LOGGER:
        return
    t = time.perf_counter()
    global_logger = _get_global_logger()
    global_logger.event_start.remote(
        timestamp=t,
        owner_name=owner_name,
        event_category=event_category,
        event_id=event_id,
    )


def event_stop(owner_name: str, event_category: str, event_id: Optional[int] = None):
    if not _ENABLE_EVENT_LOGGER:
        return
    t = time.perf_counter()
    global_logger = _get_global_logger()
    global_logger.event_stop.remote(
        timestamp=t,
        owner_name=owner_name,
        event_category=event_category,
        event_id=event_id,
    )


def initialize():
    """Initializes the global event logger.

    The global event logger should be run in detached mode
    so that different processes are able to access it.

    In initialization, if we discover that the actor
    was not cleaned up properly, we delete it and re-create it.

    If not, then we simply create the global actor.

    This function should only be called once per job.

    """
    try:
        actor = ray.get_actor("event_logger", namespace="metrics")
        logging.info("Discovered an existing event logger. Cleaning up...")
        ray.kill(actor)
    except ValueError:
        # If we fail, then the actor was never initialized.
        # This is the expected behavior
        logging.info("Could not find an existing event logger.")
        logging.info("This is expected behavior...")
        pass
    logging.info("Creating global logger...")
    EventLogger.options(
        lifetime="detached", name="event_logger", namespace="metrics"
    ).remote()


def teardown():
    ray.kill(_get_global_logger())


def summary():
    if not _ENABLE_EVENT_LOGGER:
        return
    global_logger = _get_global_logger()
    ray.get(global_logger.summary.remote())
