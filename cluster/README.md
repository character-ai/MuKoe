# Ray Cluster Setup

We support running Mukoe on both GCE and GKE.

## Pre-requisites

Please make sure that you have already built and deployed your Docker image(s) as specified in [docker](..docker/).

Please also make sure that you have a GCP project that is capable of provisioning and accessing [Google Cloud TPUs](https://cloud.google.com/tpu/).

## Running on GCE

To get started quickly, you may find it convenient to use our provided cluster definition in [mukoe.yaml](mukoe.yaml). We highly suggest getting familiar with the [Ray CLI tools](https://docs.ray.io/en/latest/cluster/cli.html).

Please make sure to replace all instances of <YOUR-PROJECT-ID> in the cluster definition with the id of your GCP project.

To create the cluster:

```
ray up -y mukoe.yaml
```

To teardown the cluster:

```
ray down -y mukoe.yaml
```

## Running on GKE
TODO