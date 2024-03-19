# Docker

We provide a Dockerfile as a reference point for creating Mukoe-compatible containers for both CPUs and TPUs.

For convenience, to build and deploy these images, you can use the following script (run this from the root `mukoe` directory):

```
$ ./scripts/build_docker.sh $PROJECT_NAME cpu|tpu|all
```

where $PROJECT_NAME is the name of your GCP project.
