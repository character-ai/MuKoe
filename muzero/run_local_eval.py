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
# to get the gif:
# brew install ImageMagick
# cd saving_dir
# convert *.jpeg screens.gif

"""Run a model on an atari env and get images"""
import os
import jax
import networks
import atari_env_loop
import config
from actor import MzActor
import specs
from PIL import Image
import time

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["JAX_PLATFORMS"] = "cpu"

# args
actor_id = 0
environment_name = "MsPacman"
ckpt_dir = "/home/wendy/ray-demo-MsPacman"
saving_dir = "/home/wendy/ray-demo-MsPacman-img"


# Set up the env
key1, key2 = jax.random.split(jax.random.PRNGKey(actor_id))

environment = atari_env_loop.make_atari_environment(environment_name)
environment_specs = specs.make_environment_spec(environment)
input_specs = atari_env_loop.get_environment_spec_for_init_network(environment_name)
extra_specs = atari_env_loop.get_extra_spec()

model_config = config.ModelConfig()
model = networks.get_model(model_config)


mz_actor = MzActor(
    network=model,
    observation_spec=input_specs,
    actor_id=actor_id,
    rng=key1,
    ckpt_dir=ckpt_dir,
    ckpt_save_interval_steps=10,
    adder=None,
    mcts_params=None,
    use_argmax=False,
    use_mcts=True,
)

print("The model loaded is ", mz_actor._latest_step)

env_loop = atari_env_loop.AtariEnvironmentLoop(environment, mz_actor)

time1 = time.time()
final_return, observations = env_loop.generate_episode()
time2 = time.time()

print("the final reburn is", final_return)

isExist = os.path.exists(saving_dir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(saving_dir)
    print("The new directory is created!")
    print(saving_dir)

for i in range(len(observations)):
    im_name = os.path.join(saving_dir, f"{i:05d}.jpeg")
    im = Image.fromarray(observations[i][:, :, :, 0])
    im.save(im_name)
