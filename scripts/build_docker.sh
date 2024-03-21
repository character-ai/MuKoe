#!/bin/bash

build_and_deploy() {
  GCR_DEST=gcr.io/$1/mukoe:$2
  if [ "$2" = "tpu" ]; then
    command="docker build -f docker/Dockerfile --network=host --build-arg tpu=true -t $GCR_DEST ."
  else
    command="docker build -f docker/Dockerfile --network=host -t $GCR_DEST ."
  fi
  echo "Building $GCR_DEST..."
  echo "Command: $command"
  $command

  echo "Built! Now pushing to GCR."
  docker push $GCR_DEST
}

# Function to display help message
show_help() {
  echo "Helper script to build and deploy Docker images."
  echo "Script usage: ./build.sh $PROJECT_NAME -cpu|tpu|all."
}

# Parse command line arguments
while getopts ":h" opt; do
  case ${opt} in
    h )
      show_help
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      show_help
      exit 1
      ;;
  esac
done

# Shift to get the remaining arguments
shift $((OPTIND -1))

# Check remaining arguments
if [ "$#" -ne 2 ]; then
  echo "Exactly two arguments required: $PROJECT_NAME cpu|tpu|all."
  show_help
  exit 1
fi

# Main logic for building
case $2 in
  tpu)
    echo "Building for tpu..."
    build_and_deploy $1 tpu 
    ;;
  cpu)
    echo "Building for cpu..."
    build_and_deploy $1 cpu
    ;;
  all)
    echo "Building for cpu and tpu..."
    build_and_deploy $1 cpu
    build_and_deploy $1 tpu
    ;;
  *)
    echo "Invalid argument: $2"
    show_help
    exit 1
    ;;
esac

