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
from typing import List, Mapping, Optional, Tuple, Any

from absl import logging
import acme_types as types
import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

import jax_types
from mcts import base as mcts_base
from mcts import l2g_mcts
from mcts import utils as mcts_utils
import networks
import adder
import utils
import checkpoint_utils
import time
import array_encode_decode
import grpc_batcher

_MAX_RETRIES = 5
_TIMEOUT_ON_FAILED_REQUEST_S = 10


def convert(x):
    return np.expand_dims(array_encode_decode.ndarray_from_bytes(x), axis=0)


class MzActor:
    """Mz actor derived from BaseMzNetwork."""

    def __init__(
        self,
        network: networks.MzNet,
        observation_spec: specs.Array,
        actor_id: int,
        rng: jax_types.PRNGKey,
        ckpt_dir: str,
        ckpt_save_interval_steps: int,
        dyna_endpoint: Optional[str] = None,
        repr_endpoint: Optional[str] = None,
        adder: Optional[adder.SequenceAdder] = None,
        mcts_params: Optional[mcts_utils.MCTSConfig] = None,
        use_argmax: bool = False,
        use_mcts: bool = True,
        total_training_steps: int = 2_000_000,
    ) -> None:
        """Init the Mz actor.

        Args:
          network: A mz_net implementation. Only the base class's interface is used.
          observation_spec: An observeration spec.
          actor_id: The index associated with this actor.
          rng: The jax random number generator key in the shape of [2].
          ckpt_dir: The checkpoint directory.
          ckpt_save_interval_steps: Interval (in steps) of checkpoint loading.
          dyna_endpoint: If provided, this represents the endpoint of the associated
            TPU server for dyna inferences.
          repr_endpoint: If provided, this represents the endpoint of the associated
            TPU server for repr inferences.
          adder: Adder to store the action sequence.
          mcts_params: Monte Carlo Tree Search parameters.
          use_argmax: True if the actor uses the argmax of logits to select an
            action.
          use_mcts: True if the actor uses mcts to search for an action. Internally,
            mcts samples from the child nodes' count probabilities.
          total_training_steps: The total number of training steps.
        """
        self.actor_id = actor_id
        self._rng = rng
        self._ckpt_save_interval_steps = ckpt_save_interval_steps
        self._ckpt_dir = ckpt_dir

        if dyna_endpoint:
            logging.info("Got dyna endpoint: %s", repr_endpoint)
            self._dyna_client = grpc_batcher.MukoeBatcherClient(
                server_address=dyna_endpoint
            )
        else:
            self._dyna_client = None
        if repr_endpoint:
            logging.info("Got repr endpoint: %s", repr_endpoint)
            self._repr_client = grpc_batcher.MukoeBatcherClient(
                server_address=repr_endpoint
            )
        else:
            self._repr_client = None

        # We split the key here because we also need a random number when sampling
        # action according to logits in _select_action_using_logits.
        self._rng, sub_key = jax.random.split(self._rng)

        self._adder = adder
        self._mcts_params = mcts_utils.get_default_mcts_params()

        if mcts_params is not None:
            self._mcts_params.update(mcts_params)
        logging.info("mcts config at step 0: %s", self._mcts_params)

        # Load latest step and model params
        self._latest_step = 0
        self._model_params_states = None
        self._finished = False
        logging.info(f"ckpt dir {ckpt_dir}")
        retries = 0
        while retries < _MAX_RETRIES:
            try:
                self._ckpt_manager = checkpoint_utils.get_ckpt_manager(
                    ckpt_dir, ckpt_save_interval_steps, create=False, use_async=False
                )
                logging.info("got ckpt manager")
                break
            except Exception as e:
                if retries == _MAX_RETRIES:
                    logging.error("Hit max retries, shutting down now.")
                    raise e
                logging.info("Error caught: %s", e)
                logging.info("waiting for 30s and retry.")
                retries += 1
                time.sleep(30)

        all_steps = self._ckpt_manager.all_steps(read=True)
        latest_step = max(all_steps) if all_steps else None
        if latest_step is None:
            latest_step = 0

        logging.info(f"need to load actor latest_ckpt_step={latest_step}")
        self._latest_step = latest_step
        retries = 0
        while retries < _MAX_RETRIES:
            try:
                restored = self._ckpt_manager.restore(latest_step)
                restored_params = restored["save_state"]["state"]
                self._model_params_states = restored_params
                break
            except Exception as e:
                if retries == _MAX_RETRIES:
                    logging.error("Hit max retries, shutting down now.")
                    raise e
                logging.info("Error caught: %s", e)
                logging.info(f"trying to load {latest_step} again")
                retries += 1
                time.sleep(30)

        self._use_argmax = use_argmax
        self._use_mcts = use_mcts

        def repr_and_pred(params, obs, dtype=jnp.float32):
            return network.apply(params, obs, dtype, method=network.repr_and_pred)

        def dyna_and_pred(params, embedding, action):
            return network.apply(
                params, embedding, action, method=network.dyna_and_pred
            )

        self._jitted_dyna_and_pred = jax.jit(dyna_and_pred, backend="cpu")
        self._jitted_repr_and_pred = jax.jit(repr_and_pred, backend="cpu")

        self._total_training_steps = total_training_steps
        if self._use_mcts:
            model_fns = mcts_base.ModelFunctions(
                repr_and_pred=self._repr_and_pred,
                dyna_and_pred=self._dyna_and_pred,
                get_legal_actions_mask=self._get_legal_actions_mask,
            )

            # A MCTS search implementation.
            self._mcts_search = l2g_mcts.L2GSearch(
                params=self._mcts_params,
                model_fns=model_fns,
                total_training_steps=self._total_training_steps,
            )

        # The policy probs and raw value of the most recent step.
        self._policy_probs = jnp.zeros(
            self._mcts_params.action_space_size, dtype=jnp.float32
        )
        self._raw_value = jnp.float32(0)

        self._metrics = {}

    def get_metrics(self) -> Mapping[str, float]:
        """Gets the metrics of the last action selection."""
        return self._metrics

    def _get_prediction_fn_output(
        self, prediction: Mapping[str, jnp.ndarray]
    ) -> mcts_base.PredictionFnOutput:
        """Gets a PredictionFnOutput from network prediction."""
        # Convert jax's DeviceArray to ndarray because it will be faster for the
        # MCTS lib to use numpy operations.
        value = np.asanyarray(prediction[networks.VALUE])
        reward = np.asanyarray(prediction[networks.REWARD])
        action_logits = np.asanyarray(prediction[networks.POLICY])
        # squeeze batch dim.
        value = np.squeeze(value, axis=0)  #  []
        reward = np.squeeze(reward, axis=0)  # []
        action_logits = np.squeeze(action_logits, axis=0)  #  [action_space_size]
        return mcts_base.PredictionFnOutput(
            value=value, reward=reward, action_logits=action_logits
        )

    def _repr_and_pred(
        self, obs: types.NestedArray
    ) -> Tuple[jnp.ndarray, mcts_base.PredictionFnOutput]:
        """Gets the representation of the observation.

        Args:
          obs: The observation's first pass stats is in [B, T, D] where D is the
            number of features.

        Returns:
          A tuple of representation in [B, embedding_dim] and PredictionFnOutput.
          The PredictionFnOutput's value in [] and action logits in [num_actions]
          are saved as numpy.ndarray.
        """
        if self._repr_client is not None:
            logging.debug("Making repr request to inference actor...")
            obs = np.asarray(obs)
            obs = array_encode_decode.ndarray_to_bytes(obs)
            received_response = False
            while not received_response:
                try:
                    repr_net_out, step = self._repr_client.send_request(data=obs)
                    if step != self._latest_step:
                        self.update()
                    self._latest_step = step
                    embedding = convert(repr_net_out["embedding"])
                    pred_out = {}
                    for k in repr_net_out:
                        if k == "embedding":
                            continue
                        else:
                            pred_out[k] = convert(repr_net_out[k])
                    pred_out = self._get_prediction_fn_output(pred_out)
                    received_response = True
                except Exception as e:
                    logging.info(
                        "Failed on repr client request with %s. Waiting for %d s.",
                        e,
                        _TIMEOUT_ON_FAILED_REQUEST_S,
                    )
                    time.sleep(_TIMEOUT_ON_FAILED_REQUEST_S)
        else:
            model_params_states = self._get_model_params_states()
            rp_net_out = self._jitted_repr_and_pred(
                {"params": model_params_states["params"]}, obs
            )
            embedding = rp_net_out[0]
            pred_out = self._get_prediction_fn_output(rp_net_out[1])
        return embedding, pred_out

    def _dyna_and_pred(
        self, embedding: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, mcts_base.PredictionFnOutput]:
        """Gets the next state & predictions from the current state & action pair.

        Args:
          embedding: The state embedding in [B, embedding_dim].
          action: A scalar value to represent the action.

        Returns:
          A tuple of the state in [B, embedding_dim] and PredictionFnOutput.
          The PredictionFnOutput's value in [] and action logits in [num_actions]
          are saved as numpy.ndarray.
        """
        action = action[jnp.newaxis,]  #  [1,]
        model_params_states = self._get_model_params_states()
        if self._dyna_client is not None:
            logging.debug("Making dyna request to inference actor...")
            # no axis dim
            embedding = array_encode_decode.ndarray_to_bytes(embedding)
            received_response = False
            while not received_response:
                try:
                    dp_net_out, step = self._dyna_client.send_request(
                        data=embedding, action=int(action)
                    )
                    embedding = convert(dp_net_out["embedding"])
                    pred_out = {}
                    for k in dp_net_out:
                        if k == "embedding" or k == "step":
                            continue
                        else:
                            pred_out[k] = convert(dp_net_out[k])
                    pred_out = self._get_prediction_fn_output(pred_out)
                    received_response = True
                except Exception as e:
                    logging.info(
                        "Failed on dyna client request with %s. Waiting for %d s...",
                        e,
                        _TIMEOUT_ON_FAILED_REQUEST_S,
                    )
                    time.sleep(_TIMEOUT_ON_FAILED_REQUEST_S)
        else:
            dp_net_out = self._jitted_dyna_and_pred(
                {"params": model_params_states["params"]}, embedding, action
            )
            embedding = dp_net_out[0]
            pred_out = self._get_prediction_fn_output(dp_net_out[1])
        return embedding, pred_out

    def update(self) -> None:
        all_steps = self._ckpt_manager.all_steps(read=True)
        latest_step = max(all_steps) if all_steps else None
        logging.info(f"actor latest_ckpt_step={latest_step}")
        if latest_step:
            count_try = 0
            if self._latest_step == latest_step:
                return True
            while True:
                if count_try > 3:
                    return False
                try:
                    restored = self._ckpt_manager.restore(latest_step)
                    restored_params = restored["save_state"]["state"]
                    restored_step = restored_params["step"]
                    logging.info(f"actor restored_ckpt_step={restored_step}")
                    self._latest_step = latest_step
                    self._model_params_states = restored_params
                    if self._latest_step - 1 >= self._total_training_steps:
                        self._finished = True
                    return True
                except Exception:
                    count_try += 1
                    logging.info("waiting for 30s and retry updating actor.")
                    time.sleep(30)
        else:
            return False

    def _get_legal_actions_mask(self, actions_sequence: List[Any]) -> np.ndarray:
        """Gets the mask for legal actions.

        Args:
          actions_sequence: The action history so far.

        Returns:
          The legal action mask in [action_space_size]. 1 means valid action, 0
            means invalid action.
        """
        del actions_sequence
        # All actions are legal.
        return np.ones(self._mcts_params.action_space_size, dtype=np.int32)

    def _get_training_steps(self) -> int:
        """Gets the number of training steps."""
        return self._latest_step

    def finished(self) -> bool:
        return self._finished

    def select_action(
        self, obs: types.NestedArray, is_training: bool = True
    ) -> types.NestedArray:
        """Select an action according to MCTS.

        Args:
          obs: The observation. Its first pass stats is in [T, D] where D is the
            number of features.
          is_training: If it is for training or eval.

        Returns:
          The selected action in [].
        """
        batched_obs = utils.add_batch_dim(obs)  # [B, T, D]. B = 1
        if self._use_mcts:
            training_steps = self._get_training_steps()

            # Use the normalized visit counts as the policy probs.
            action, self._policy_probs, self._metrics = self._mcts_search.gen_action(
                observation=batched_obs, training_steps=training_steps
            )
            self._raw_value = self._metrics[networks.RAW_VALUE]
        else:
            action, self._policy_probs = self._select_action_using_logits(
                obs=batched_obs
            )
            self._raw_value = 0.0
        if not is_training:
            # if not training, use greedy
            action = jnp.argmax(self._policy_probs)
        return action

    def _select_action_using_logits(self, obs: types.NestedArray) -> types.NestedArray:
        """Selects an actions using the network logits.

        Args:
          obs: Observation which first pass stats in [B, T, D] where D is the num of
            features and B = 1.

        Returns:
          A tuple of the selected action and the action probabilities.
        """
        # Forward-pass.
        unused_embedding, pred_out = self._repr_and_pred(obs=obs)

        if self._use_argmax:
            action = jnp.argmax(pred_out.action_logits)  #  logits in [action space]
        else:
            self._rng, sub_key = jax.random.split(self._rng)
            action = jax.random.categorical(key=sub_key, logits=pred_out.action_logits)

        # Remove batch dimension
        action = jnp.squeeze(action)

        # probs squeeze to shape [Action space size,]
        probs = jnp.squeeze(jax.nn.softmax(pred_out.action_logits))
        return action, probs

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        """Observes the first step and adds to both data and reanalysis buffers."""
        self._reset()
        if self._adder:
            self._adder.add_first(timestep)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ) -> None:
        """Observes an action & next step pair, adds them to the replay buffer."""
        if self._adder:
            extras = {}
            extras[networks.POLICY_PROBS] = self._policy_probs
            extras[networks.NETWORK_STEPS] = jnp.asarray(self._get_training_steps())
            if self._use_mcts:
                extras[networks.RAW_VALUE] = jnp.asarray(self._raw_value)
            else:
                extras[networks.RAW_VALUE] = jnp.asarray(0, jnp.float32)
            self._adder.add(action, next_timestep, extras=extras)

    def _reset(self) -> None:
        """Resets the actor."""
        self._policy_probs = jnp.zeros(
            self._mcts_params.action_space_size, dtype=jnp.float32
        )
        self._raw_value = jnp.float32(0)

    def _get_model_params_states(self):
        """Returns the mz model params."""
        return self._model_params_states
