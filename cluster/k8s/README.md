# Running MuKoe on GKE

Running MuKoe on GKE utilizes KubeRay, and requires the steps of:
- Create the GKE cluster
- Create Ray-related resources
- Deploy KubeRay and other relevant CRDs/services
- Deploy the workload.


## Environment setup
We suggest using environment variables to simplify the cluster creation steps:

```
CLUSTER_NAME=mukoe-cluster
PROJECT=<my-project-name>
ZONE=us-central2-b
SVC_ACCOUNT=ray-sa@${PROJECT}.iam.gserviceaccount.com
```

Notes:
- Make sure to change `PROJECT` to your project
- We use `us-central2-b` as we primarily developed on TPUv4. Please see [TPU regions and zones](https://cloud.google.com/tpu/docs/regions-zones) for other available TPU zones!

## Creating the GKE cluster

```
gcloud container clusters create ${CLUSTER_NAME} \
  --project=${PROJECT} \
  --workload-pool=${PROJECT}.svc.id.goog \
  --scopes=cloud-platform \
  --enable-ip-alias \
  --zone=${ZONE} \
  --release-channel=regular \
  --enable-tpu
```

Notes:
- We use `cloud-platform` scope to enable pods deployed in our cluster to interact with other services, like GCS
- We use the `regular` release channel to access GKE version >=1.28 which was needed at the time of writing.
- We set `--enable-tpu` as MuKoe runs on TPUs.

## Creating GKE Node Pools

MuKoe utilizes several accelerator types:
- CPUs (`n2-standard-16`) for actors and for the Ray head node
- Single-host TPUs (`v4-8`) for TPU inference actors, learner, and reverb server.
    - Optionally, multi-host TPUs (i.e. `v4-16`) can be used for the learner.

Notes:
- "Single-host" TPUs, i.e. TPU VMs typically with at most 4 chips.
    - "Multi-host" TPUs, on the other hand, refer to TPU pod slices with more than one single host TPU. See [TPUs in GKE introduction](https://cloud.google.com/tpu/docs/tpus-in-gke) for more information.
- The Reverb server does not use the TPU accelerators, but does require a large amount of memory. You could also replace the v4-8 with a high memory CPU option.
- If you want to change the accelerator type, this may require downstream changes in both the Ray cluster manifest and the Ray-side code.

We outline the nodepool configurations we used in MuKoe. 

### Create the CPU nodepool

Create a CPU nodepool with 100 `n1-standard-16`s:

```
gcloud container node-pools create cpu-pool \
    --cluster=${CLUSTER_NAME} \
    --machine-type=n1-standard-16 \
    --num-nodes=100 \
    --zone=${ZONE} \
    --service-account=${SVC_ACCOUNT} \
    --scopes=cloud-platform \
    --project=${PROJECT}
```

### Create the TPU single-host nodepool

Create a TPU nodepool with 12 v4-8s (`ct4p-hightpu-4t`):

```
gcloud container node-pools create tpu-pool \
  --location=${ZONE} \
  --project=${PROJECT} \
  --cluster=${CLUSTER_NAME} \
  --service-account=${SVC_ACCOUNT} \
  --node-locations=${ZONE} \
  --scopes=cloud-platform \
  --machine-type=ct4p-hightpu-4t \
  --num-nodes=12
```

gcloud container node-pools create tpu-pool-2 \
  --location=${ZONE} \
  --project=${PROJECT} \
  --cluster=${CLUSTER_NAME} \
  --service-account=${SVC_ACCOUNT} \
  --node-locations=${ZONE} \
  --scopes=cloud-platform \
  --machine-type=ct4p-hightpu-4t \
  --num-nodes=2

Note: This assumes 10 will be used for TPU inference, 1 for the learner, 1 for the reverb server.

### (Optional) Create a TPU multi-host nodepool

Create a TPU nodepool with one v4-16 (`ct4p-hightpu-4t` with topology 2x2x2):

```
gcloud container node-pools create tpu-multi \
  --location=${ZONE} \
  --project=${PROJECT} \
  --cluster=${CLUSTER_NAME} \
  --service-account=${SVC_ACCOUNT} \
  --node-locations=${ZONE} \
  --scopes=cloud-platform \
  --machine-type=ct4p-hightpu-4t  \
  --tpu-topology=2x2x2 \
  --num-nodes=2
```

## GKE config sanity check
Before moving on, sanity check that your GKE cluster and node pools have version >= 1.28. If this isn't the case, you may see [permissions errors with TPU chip access](https://cloud.google.com/kubernetes-engine/docs/troubleshooting/tpus). You can upgrade with these command:
```
# Cluster upgrade
gcloud container clusters upgrade $CLUSTER_NAME --project=${PROJECT} --zone=${ZONE} --master --cluster-version=1.28.3-gke.1286000

# Nodepool upgrade
gcloud container clusters upgrade $CLUSTER_NAME --project=${PROJECT} --zone=${ZONE} --node-pool=$NODEPOOL_NAME --cluster-version=1.28.3-gke.1286000
```

where $NODEPOOL_NAME will either be `cpu-pool`, `tpu-pool` and/or `tpu-multi`.

Connect to your cluster:
```
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT}
```

Also ensure that you have built the Docker images as described in [docker](../docker).

## Deploying Relevant KubeRay Resources
MuKoe uses [Ray](ray.io) for scaling out resources in a distributed manner. The recommended path for running Ray on GKE is through [KubeRay](https://github.com/ray-project/kuberay). For TPUs and KubeRay, there are additional steps required as well to deploy a [TPU-specific webhook](https://github.com/GoogleCloudPlatform/ai-on-gke/tree/main/applications/ray/kuberay-tpu-webhook). For simplicity, we provide the exact commands that we use.

First, deploy the KubeRay operator:

```
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0-rc.0
```

If you see an issue with this, you may need to upgrade from helm:

```
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0-rc.0
```

After the `kuberay-operator` is deployed, you will also need to deploy the `kuberay-tpu-webhook`. For the most up-to-date instructions, we suggest referring to the [examples hosted by GKE](https://github.com/GoogleCloudPlatform/ai-on-gke/tree/main/applications/ray/kuberay-tpu-webhook). For simplicity, we also have the commands listed here:

```
# Deploys the cert manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Deploys reflector
kubectl apply -f https://github.com/emberstack/kubernetes-reflector/releases/latest/download/reflector.yaml

# Deploys the webhook
kubectl apply -f tpu-webhook/deployments/

# Deploys certs
kubectl apply -f tpu-webhook/certs/

```

## Deploying the Ray cluster
To deploy the Ray cluster, we've provided a sample YAML file at [cluster.yaml](cluster.yaml). Ensure that you replace all instances or <YOUR-GCP-PROJECT-ID> with your project ID! E.g.:

```
sed -i "s/<YOUR-GCP-PROJECT-ID>/${PROJECT}/g" cluster.yaml
```

You are now able to deploy your KubeRay cluster:

```
kubectl apply -f cluster.yaml
```

Additionally set the

```
kubectl annotate serviceaccount default iam.gke.io/gcp-service-account=ray-sa@tpu-vm-gke-testing.iam.gserviceaccount.com
```

To run a quick sanity check, you can attach to the head node via:

```
./scripts/attach_head_k8s.sh
```


## Teardown

To teardown the Ray cluster:
```
kubectl delete -f cluster.yaml
```

To teardown the TPU webhook:
```
# Deletes cert manager
kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Deletes reflector
kubectl delete -f https://github.com/emberstack/kubernetes-reflector/releases/latest/download/reflector.yaml

# Deletes the webhook
kubectl delete -f tpu-webhook/deployments/

# Deletes certs
kubectl delete -f tpu-webhook/certs/
```

To teardown KubeRay:
```
helm uninstall kuberay-operator
```




