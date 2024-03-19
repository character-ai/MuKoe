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
"""Tests for ray_inference_state.py

Run this with:
python3 -m pytest -v -rP tests/test_ray_inference_state.py
"""
import unittest
from unittest.mock import Mock, patch
import ray

from ray_inference_state import InferenceStateHandler


class TestInferenceStateHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    @patch("ray_inference_state._get_cluster_inference_resource_ids")
    @patch("ray_inference_state._get_available_inference_resource_ids")
    def setUp(self, mock_available_resources, mock_cluster_resources):
        # Define the mock return values for resource ID functions
        mock_cluster_resources.return_value = {0, 1}
        mock_available_resources.return_value = {0, 1}

        self.handler = InferenceStateHandler(
            num_dyna_inferers=2,
            num_repr_inferers=2,
            num_actors_per_repr=3,
            num_actors_per_dyna=3,
            repr_resource_base="inference_v4_8_repr",
            dyna_resource_base="inference_v4_8_dyna",
        )

    def test_register_and_get_endpoint(self):
        mock_actor = Mock()
        endpoint = "127.0.0.1:8000"
        model = "repr"
        actor_id = self.handler.request_id(model)

        self.handler.register_inferer(model, mock_actor, endpoint, actor_id)
        retrieved_endpoint = self.handler.register_and_get_endpoint(model, actor_id)

        self.assertEqual(endpoint, retrieved_endpoint)

    def test_unregister_inferer(self):
        mock_actor = Mock()
        model = "dyna"
        actor_id = self.handler.request_id(model)

        self.handler.register_inferer(model, mock_actor, "127.0.0.1:8001", actor_id)
        self.handler.unregister_inferer(mock_actor)

        with self.assertRaises(AssertionError):
            self.handler.register_and_get_endpoint(model, actor_id)

    def test_get_associated_actor_ids(self):
        mock_actor = Mock()
        model = "repr"
        actor_id = self.handler.request_id(model)
        expected_ids = [0, 1]
        self.handler.register_inferer(model, mock_actor, "127.0.0.1:8002", actor_id)

        for i in expected_ids:
            self.handler.register_and_get_endpoint(model=model, actor_id=i)

        associated_ids = self.handler.get_associated_actor_ids(mock_actor)

        self.assertEqual(associated_ids, expected_ids)

    def test_get_inference_handle(self):
        mock_actor = Mock()
        model = "dyna"
        actor_id = self.handler.request_id(model)
        self.handler.register_inferer(model, mock_actor, "127.0.0.1:8003", actor_id)

        retrieved_actor = self.handler.get_inference_handle(model, actor_id)
        self.assertEqual(retrieved_actor, mock_actor)

        # Test retrieval with an incorrect actor_id
        incorrect_actor = self.handler.get_inference_handle(model, 999)
        self.assertIsNone(incorrect_actor)

    def test_allocate_id_with_no_available_ids(self):
        # Setup the handler to have no available IDs for a specific model
        handler = InferenceStateHandler(
            num_dyna_inferers=0,
            num_repr_inferers=0,
            num_actors_per_repr=0,
            num_actors_per_dyna=0,
            repr_resource_base="inference_v4_8_repr",
            dyna_resource_base="inference_v4_8_dyna",
        )

        # Test _allocate_id to ensure it raises an AssertionError when no IDs are available
        with self.assertRaises(AssertionError):
            handler._allocate_id(model="repr")

    def test_register_inferer_with_invalid_model(self):
        mock_actor = Mock()
        endpoint = "127.0.0.1:8000"
        actor_id = self.handler.request_id(model="repr")

        # Attempt to register an inferer with an invalid model type
        with self.assertRaises(ValueError):
            self.handler.register_inferer(
                model="invalid_model",
                actor=mock_actor,
                endpoint=endpoint,
                inferer_id=actor_id,
            )

    def test_get_associated_actor_ids_with_unregistered_actor(self):
        mock_actor = Mock()

        # Attempt to get associated actor IDs for an unregistered actor
        with self.assertRaises(AssertionError):
            self.handler.get_associated_actor_ids(actor=mock_actor)


if __name__ == "__main__":
    unittest.main()
