# MuKoe

MuKoe is an implementation of MuZero using Ray as the distributed orchestrator on GKE. We provide examples with Atari.

## Environment setup
To set up the software environment used in MuKoe, feel free to utilize our provided [docker](docker/) files and scripts to build both the CPU and TPU docker images:

```
./scripts/build_docker.sh $GCP_PROJECT_ID all
```

Note that this will upload the image to Google Cloud Container Registry (GCR): gcr.io/$GCP_PROJECT_ID/ray-muzero:cpu and  gcr.io/$GCP_PROJECT_ID/ray-muzero:tpu.

## Setting up the Cluster
We provide instructions for setting up your Ray cluster in [cluster/](cluster/). To quickly get started using GCE VMs, you can spin up your cluster with:

```
ray up -y cluster/mukoe.yaml
```

To teardown the cluster, you can do so with

```
ray down -y cluster/mukoe.yaml
```

## Instructions to run MuKoe

Once your Ray cluster is setup, you can run MuKoe as a Ray job.

### Required and Expected Resources

Our implementation of MuKoe has specific hardware requirements:
* The learner should be a more powerful TPU accelerator, that learns the model via sgd.
* the replay buffer should be a node with at least 300G to 500G ram:
* the remote handler is a standard cpu node, it does not have to do real work but to act as an intermediate handler.
* The actors depend on if you are using CPU inference or batched TPU inference. We illustrate the set up individually next. 

Once your cluster is set up, you should be able to check that the resources in the Ray cluster look something like the following (run this from the head node):

```
$ ray status

Resources
---------------------------------------------------------------
Usage:
 0.0/13056.0 CPU
 0.0/8.0 TPU
 0.0/800.0 actor
 0.0/1.0 inference_cpu_handler
 0B/59.93TiB memory
 0B/18.25TiB object_store_memory
 0.0/1.0 replay_buffer
 0.0/1.0 learner

Demands:
 (no resource demands)
```

### Running the job with CPU actors

To run on CPU, make sure your `ray status` show the following resources. Specifically, we need at least 16 cpu cores per actor. For instance, if your cpu node has 64 cores, then assign 4 actor resources to in your cluster definition:

```
Resources
---------------------------------------------------------------
Usage:
 1.0/13056.0 CPU
 0.0/800.0 actor
 0.0/8.0 inference_cpu_handler
 0B/59.93TiB memory
 0B/18.25TiB object_store_memory
 1.0/1.0 replay_buffer
 0.0/2.0 learner

Demands:
 (no resource demands)
```

To start the CPU job, launch `ray_main.py` with Ray API:
```
ray job submit --working-dir muzero -- RAY_DEUP_LOGS=0 python ray_main.py \
    --ckpt_dir ${CKPT_DIR} \
    --save_dir ${SAVE_DIR} \
    --tensorboard_dir ${TB_DIR} \
    --reverb_dir ${REVERB_DIR} \
    --num_actors ${NUM_ACTORS} \
    --core_per_task ${CORES_PER_TASK} \
    --environment ${ENVIRONMENT} \
    --inference_node cpu
```

If developing on a machine separate from the Ray Head node, you may find it useful to create a port forward (port 8265) to the Ray head node and set `RAY_ADDRESS=http://127.0.0.1:8265`, then use the [Ray Job CLI](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html).

Sample commands:

(from terminal 1):

```
$ ray attach -p 8265 cluster/mukoe.yaml
```

(from terminal 2):

```
$ export RAY_ADDRESS=http://127.0.0.1:8265
$ ray job submit ...
```

We have two scripts: `ray_main.py` and `ray_benchmark.py` within the `muzero` folder that can also be
run using the ray Job API.

### Run on TPU

WIP

# Contributions

### Character.AI:

- Wendy Shang: main contributor, MuZero algorithm implementation and ray integration. 

### Google:

- Allen Wang: main contributor for Ray core integrations, TPU batchers, cluster setup
- Jingxin Ye, Richard Liu, Han Gao, Ryan O'Leary: collaborators, prototyping and GKE / Kuberay infra support


# Acknowledgement 
This work is built upon a collection of previous efforts, by many predecessors including:
- Google Deepmind ACME Team `https://github.com/google-deepmind/acme` 
- Google Research Zurich Learn to Google Project `https://github.com/google-research/google-research/tree/master/muzero`
- Google Deepmind Applied (Miaosen Wang, Flora Xue, Chenjie Gu) `https://arxiv.org/abs/2202.06626`
- Character.AI infra support: Stephen Roller, Myle Ott 

### Project Name: MuKoe
The name Mu is derived from the algorithm we have re-implemented, known as MuZero. This title encompasses various nuanced meanings, as elaborated in [author note](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/). Notably, in Japanese, the kanji Mu, 夢, translates to "dream."

The component Ko, as in Kou 光, signifies "light," symbolizing our orchestrator framework named Ray, akin to a ray of light.

Lastly, Koe simultaneously denotes the kanji 声, meaning "voice," and references the Kubernetes cluster.


## License
Copyright 2024 Character Technologies Inc. and Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.