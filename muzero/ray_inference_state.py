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
import ray
from typing import Iterable, List, Optional, Set
import logging


def _get_cluster_inference_resource_ids(resource_base: str) -> Set[int]:
    """Returns the set of cluster resource IDs based on a base resource string.

    Args:
        resource_base (str): The resource string base. Example:
            "inference_v4_8_repr" or "inference_v4_8_dyna".
            This also assumes there is a corresponding "CPU" resource string
            associated with each Ray node, i.e. inference_repr_cpu_handler_0.
    """
    cluster_resources = ray.cluster_resources()
    return sorted(
        [
            int(res[len(resource_base) + 1])
            for res in cluster_resources
            if res.startswith(resource_base)
        ]
    )


def _get_available_inference_resource_ids(resource_base: str) -> Set[int]:
    """Returns the set of cluster resource IDs based on a base resource string.

    Args:
        resource_base (str): The resource string base. Example:
            "inference_v4_8_repr" or "inference_v4_8_dyna".
            This also assumes there is a corresponding "CPU" resource string
            associated with each Ray node, i.e. inference_repr_cpu_handler_0.
    """
    available_resources = ray.available_resources()
    return sorted(
        [
            int(res[len(resource_base) + 1])
            for res in available_resources
            if res.startswith(resource_base)
        ]
    )


class InferenceStateHandler:
    """
    Manages the state and mappings between inferers ('repr' and 'dyna') and Muzero actors.
    """

    def __init__(
        self,
        num_dyna_inferers: int,
        num_repr_inferers: int,
        num_actors_per_repr: int,
        num_actors_per_dyna: int,
        repr_resource_base: str,
        dyna_resource_base: str,
    ):
        """
        Initializes the InferenceStateHandler with the given configuration.

        Args:
            num_dyna_inferers (int):   The number of dyna inferencers requested by the program.
            num_repr_inferers (int):   The number of repr inferencers requested by the program.
            num_actors_per_repr (int): The number of Muzero actors associated with each 'repr' inferer.
            num_actors_per_dyna (int): The number of Muzero actors associated with each 'dyna' inferer.
            repr_resource_base (str):  The base string representing a repr resource.
            dyna_resource_base (str):  The base string representing a dyna resource.
        """
        self.num_dyna_inferers = num_dyna_inferers
        self.num_repr_inferers = num_repr_inferers

        self.repr_resource_base = repr_resource_base
        self.dyna_resource_base = dyna_resource_base

        self.num_actors_per_repr = num_actors_per_repr
        self.num_actors_per_dyna = num_actors_per_dyna

        self.available_repr_ids = set(
            _get_cluster_inference_resource_ids(resource_base=repr_resource_base)
        )
        self.available_dyna_ids = set(
            _get_cluster_inference_resource_ids(resource_base=dyna_resource_base)
        )
        logging.info(
            "Detected %d available repr resources with IDs %s",
            len(self.available_repr_ids),
            self.available_repr_ids,
        )
        logging.info(
            "Detected %d available dyna resources with IDs %s",
            len(self.available_dyna_ids),
            self.available_dyna_ids,
        )

        self.repr_endpoints = [None] * len(self.available_repr_ids)
        self.dyna_endpoints = [None] * len(self.available_dyna_ids)
        self.model_actor_map = {}
        # *_assigned_actors maintains the mapping between the repr/dyna inferers
        # and its associated actor IDs.
        self.repr_assigned_actors = [[] for _ in range(len(self.available_repr_ids))]
        self.dyna_assigned_actors = [[] for _ in range(len(self.available_dyna_ids))]
        self.inferer_handle_to_id = {}
        self.id_to_actor_handle = {}
        self.id_to_actor_handle["repr"] = {}
        self.id_to_actor_handle["dyna"] = {}

    def _allocate_id(self, model: str) -> int:
        """Allocates an ID for a new actor, reusing available slots if necessary.

        While the state manager handles its own bookkeeping for registered IDs, we
        might have scenarios where there are spare resources that can be hot swapped
        in case a previous resource failed.

        To take advantage of spares, the logic in _pick_id is to compare the list of
        tracked available IDs and compare to the resources that Ray knows is available
        and prioritize those IDs that are available from Ray's perspective.

        In case we cannot find an ID that's available from Ray's perspective, this means
        that there are no spares and we should instead re-use a resource that is
        recovering.

        Args:
            model(str): The type of the model to allocate an ID for, repr or dyna.

        Raises:
            ValueError: in case an incorrect model was provided.
            AssertionError: in case there are no detected IDs that can be allocated.

        """

        def _pick_id(resource_base: str, available_ids: Set[int]):
            if not available_ids:
                raise AssertionError(
                    "There are no remaining available IDs, meaning that we have requested "
                    "more resources than available in the cluster. This should not happen."
                )

            # If there is an available resource from Ray's perspective, this is a spare.
            # Prioritize this.
            ray_available_ids = _get_available_inference_resource_ids(
                resource_base=resource_base
            )
            for available_id in available_ids:
                if available_id in ray_available_ids:
                    available_ids.remove(available_id)
                    return available_id
            # If we've reached this point, then there are no spares.
            return available_ids.pop()

        if model == "repr":
            return _pick_id(
                resource_base=self.repr_resource_base,
                available_ids=self.available_repr_ids,
            )
        elif model == "dyna":
            return _pick_id(
                resource_base=self.dyna_resource_base,
                available_ids=self.available_dyna_ids,
            )
        else:
            raise ValueError(
                f"Invalid model type provided: {model}. Expected 'repr' or 'dyna'."
            )

    def _release_id(self, model: str, actor_id: int):
        """Releases an ID, marking it as available for reuse."""
        if model == "repr":
            self.available_repr_ids.add(actor_id)
        elif model == "dyna":
            self.available_dyna_ids.add(actor_id)

    def request_id(self, model: str) -> int:
        """Requests an ID for a given inferer.

        This should be used in conjunction with register_inferer, i.e.:

        inferer_id = state.request_id(model="repr")
        inferer = # create Repr inferer...
        state.register_inferer(model="repr", actor=inferer, endpoint=endpoint, inferer_id=inferer_id)

        Raises ValueError if invalid model type is provided.

        """
        return self._allocate_id(model=model)

    def register_inferer(
        self, model: str, actor: ray.actor.ActorHandle, endpoint: str, inferer_id: int
    ):
        """Registers a new inferer, updating the mappings and endpoint lists.

        Args:
            model (str): The model type of the actor ("repr" or "dyna").
            actor (ray.actor.ActorHandle): The Ray actor handle of the inferer.
            endpoint (str): The gRPC endpoint of the inferer.
            inferer_id (int): The index of the inferer.

        Raises a ValueError if the provided model is invalid.
        """
        if model == "repr":
            self.repr_endpoints[inferer_id] = endpoint
        elif model == "dyna":
            self.dyna_endpoints[inferer_id] = endpoint
        else:
            raise ValueError(
                f"Invalid model type {model} provided. Expected dyna or repr."
            )
        self.inferer_handle_to_id[actor] = (model, inferer_id)
        self.id_to_actor_handle[model][inferer_id] = actor

    def unregister_inferer(self, actor: ray.actor.ActorHandle):
        """Unregisters an inferer, freeing its ID for potential reuse.

        Args:
            actor (ray.actor.ActorHandle): The Ray actor handle of the inferer to be unregistered.
        """
        model, actor_id = self.inferer_handle_to_id.pop(actor, (None, None))
        del self.id_to_actor_handle[model][actor_id]
        if model and actor_id is not None:
            self._release_id(model, actor_id)
            if model == "repr":
                self.repr_endpoints[actor_id] = None
                self.repr_assigned_actors[actor_id].clear()
            else:
                self.dyna_endpoints[actor_id] = None
                self.dyna_assigned_actors[actor_id].clear()

    def register_and_get_endpoint(self, model: str, actor_id: int) -> str:
        """Retrieves the endpoint for an inferer based on the Muzero actor ID and model type.

        Args:
            model (str): The model type ("repr" or "dyna").
            actor_id (int): The ID of the Muzero actor.

        Returns:
            str: The endpoint of the associated inferer.

        Raises AssertionError if there is no assignable endpoint.
        """

        def _find_endpoint_and_register_actor(
            assigned_actor_list: Iterable[Iterable[int]],
            endpoints_list: Iterable[str],
            num_actors_per_inferer: int,
            actor_id: int,
            model: str,
        ) -> int:
            inferer_id = None
            for i, bucket in enumerate(assigned_actor_list):
                # If the actor was already registered, i.e. when re-creating a failed actor,
                # then no need to re-register.
                if actor_id in bucket:
                    logging.info(
                        "Note: actor %d  was already registered to %d", actor_id, i
                    )
                    inferer_id = i
                    break
                # A valid bucket is one where the endpoint has been registered,
                # and there is space in the assigned actor list.
                if endpoints_list[i] and len(bucket) < num_actors_per_inferer:
                    logging.info(
                        "Registering actor %d to bucket inferer %d", actor_id, i
                    )
                    inferer_id = i
                    assigned_actor_list[i].append(actor_id)
                    break
            if inferer_id is None:
                raise AssertionError(
                    f"Could not detect an assignable endpoint for actor {actor_id} and model {model}."
                    "This is not expected behavior."
                )
            return endpoints_list[inferer_id]

        if model == "repr":
            if self.num_repr_inferers == 0:
                return None
            return _find_endpoint_and_register_actor(
                assigned_actor_list=self.repr_assigned_actors,
                endpoints_list=self.repr_endpoints,
                num_actors_per_inferer=self.num_actors_per_repr,
                actor_id=actor_id,
                model=model,
            )
        elif model == "dyna":
            if self.num_dyna_inferers == 0:
                return None
            return _find_endpoint_and_register_actor(
                assigned_actor_list=self.dyna_assigned_actors,
                endpoints_list=self.dyna_endpoints,
                num_actors_per_inferer=self.num_actors_per_dyna,
                actor_id=actor_id,
                model=model,
            )

        raise AssertionError(
            "No endpoint was available. This is not expected behavior."
        )

    def get_associated_actor_ids(self, actor: ray.actor.ActorHandle) -> List[int]:
        """Retrieves the list of Muzero actor IDs associated with a given inferer handle.

        Args:
            actor (ray.actor.ActorHandle): The Ray actor handle of the inferer.

        Returns:
            List[int]: A list of associated Muzero actor IDs, or an empty list if the actor is not found.

        Raises AssertionError in case the actor was never registered.
        """
        model, inferer_id = self.inferer_handle_to_id.get(actor, (None, None))
        if model == "repr":
            return self.repr_assigned_actors[inferer_id]
        elif model == "dyna":
            return self.dyna_assigned_actors[inferer_id]
        raise AssertionError("The provided actor was not registered ")

    def model_type(self, actor: ray.actor.ActorHandle) -> Optional[str]:
        """Returns the model type of the given actor if registered."""
        model, _ = self.inferer_handle_to_id.get(actor, (None, None))
        return model

    def get_inference_handle(
        self, model: str, actor_id: int
    ) -> Optional[ray.actor.ActorHandle]:
        """Returns the ray inferer based on model name and index.

        Returns None if the handle was not registered.
        """
        id_to_actor_handle = self.id_to_actor_handle.get(model, None)
        if id_to_actor_handle:
            return id_to_actor_handle.get(actor_id, None)
        return None
