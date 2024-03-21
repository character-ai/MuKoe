# Ray Cluster Setup

We support running Mukoe on both GCE and GKE.

## Pre-requisites

For either approach, please make sure that you have already built and deployed your Docker image(s) as specified in [docker](..docker/).

Please also make sure that you have a GCP project that is capable of provisioning and accessing [Google Cloud TPUs](https://cloud.google.com/tpu/).

Running MuKoe on VMs can be useful to quickly get started when prototyping at small scale, but ultimately we recommend running the full MuKoe e2e experience with GKE.

## Running on GCE
Please refer to [vms](vms/).

## Running on GKE
Please refer to [k8s](k8s/).
