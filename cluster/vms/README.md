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