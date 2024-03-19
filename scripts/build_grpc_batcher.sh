#!/bin/bash

# Default action is to build e2e
MODE="build_e2e"

# Function to show help
show_help() {
    echo "Usage: $0 [mode]"
    echo "Modes:"
    echo "  help                  Print this message."
    echo "  build_e2e             Build the entire Python wheel in a one-off Docker container."
    echo "  build_py_interactive  Build the entire Python wheel in a persistent Docker container."
    echo "  test_py_interactive   Build and test the Python wheel in an interactive Docker container."
    echo "  build_cpp_interactive Build the C++ components in an interactive Docker container."
    echo "  teardown              Tear down any existing Docker containers created by this script."
}

# Parse command-line arguments
if [[ "$#" -gt 0 ]]; then
    MODE=$1
fi

# Function to check if the container exists and build if it doesn't
check_and_build_container() {
    CONTAINER_EXISTS=$(docker ps -a --format '{{.Names}}' | grep -w grpc_batcher_it)
    if [ -z "$CONTAINER_EXISTS" ]; then
        echo "Building grpc_batcher_base..."
        docker build -t grpc_batcher_builder_base -f docker/Dockerfile.batcher-base .
        docker run -d -it -v $(pwd)/grpc_batcher:/grpc_batcher --name grpc_batcher_it grpc_batcher_builder_base
    else
        echo "Container grpc_batcher_it already exists, re-using..."
    fi
}

case $MODE in
    help)
        show_help
        ;;
    build_e2e)
        echo "Building the wheel e2e."
        docker build -t grpc_batcher_base -f docker/Dockerfile.batcher-base .
        docker build -t grpc_batcher_builder -f docker/Dockerfile.batcher .
        echo "Container built!"
        docker create --name grpc_batcher grpc_batcher_builder "/bin/bash" 
        docker cp grpc_batcher:/grpc_batcher/grpc_batcher-0.0.1-cp310-cp310-linux_x86_64.whl grpc_batcher/dist/
        echo "Wheel copied to grpc_batcher/dist/"
        docker stop grpc_batcher
        docker rm -f grpc_batcher
        ;;
    test_py_interactive)
        echo "Testing Python wheel interactively"
        check_and_build_container
        docker exec -it grpc_batcher_it /bin/bash -c "pip install --force-reinstall --verbose . && python3 test/test_grpc_batcher.py"
        ;;
    build_py_interactive)
        echo "Building Python wheel interactively"
        check_and_build_container
        docker exec -it grpc_batcher_it /bin/bash -c "python3 setup.py bdist_wheel"
        ;;
    build_cpp_interactive)
        check_and_build_container
        docker exec -it grpc_batcher_it /bin/bash -c "cd /grpc_batcher_build && cmake /grpc_batcher/cpp && make all -j 16"
        ;;
    teardown)
        docker stop grpc_batcher_it 2>/dev/null
        docker rm -f grpc_batcher_it 2>/dev/null
        echo "Containers torn down."
        ;;
    *)
        echo "Unknown mode: $MODE"
        show_help
        exit 1
        ;;
esac
