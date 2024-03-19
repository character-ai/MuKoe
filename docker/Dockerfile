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
FROM ubuntu:22.04

USER root

ARG tpu

RUN apt-get upgrade -y && apt-get update -y && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    coreutils \
    rsync \
    openssh-client \
    curl \
    sudo \
    wget

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN pip install py-spy memray

RUN useradd -m mukoe && echo "mukoe:docker" | chpasswd && adduser mukoe sudo
#RUN useradd -m mukoe && adduser mukoe sudo
USER mukoe
WORKDIR /home/mukoe
ENV TPU=$tpu

RUN pip install cryptography
RUN pip install google-api-python-client google-cloud-storage
RUN pip install dm-reverb[tensorflow]==0.13.0
RUN pip install ipython
RUN pip install dm-env==1.6
RUN pip install 'gym==0.25.0'
RUN pip install gym[atari]==0.25.0
RUN pip install gym[accept-rom-license]==0.25.0
RUN pip install chex==0.1.82
RUN pip install rlax==0.1.6
RUN pip install numpy==1.23.5
RUN pip install pydantic==1.10.9
RUN pip install flax==0.6.11
RUN pip install Pillow==10.1.0
RUN pip install ml-collections==0.1.1
RUN pip install ml-dtypes==0.2.0
RUN pip install tensorboardx==2.6.2.2
RUN pip install orbax-checkpoint==0.5.2
RUN pip install tensorflow-cpu==2.15.0
RUN pip install jaxlib==0.4.14
RUN if [ -z "$TPU" ] ; then \
        pip install "jax[cpu]==0.4.14" ; \
    else \
        pip install "jax[tpu]==0.4.14" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ; \
    fi
RUN pip install ray[default]==2.9.0

COPY grpc_batcher/dist/grpc_batcher-0.0.1-cp310-cp310-linux_x86_64.whl .
RUN pip install grpc_batcher-0.0.1-cp310-cp310-linux_x86_64.whl
ENV LD_LIBRARY_PATH=/home/mukoe/.local/lib/python3.10/site-packages
ENV PATH="/home/mukoe/.local/bin:/home/mukoe/.local/usr/bin:$PATH"
#RUN sudo chown root:root `which py-spy` && sudo chmod u+s `which py-spy`
#RUN sudo env "PATH=$PATH"

#ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages

ENTRYPOINT ["/bin/bash"]