# GRPC Batcher

We have implemented a simple CPP<>Python gRPC batcher for Mukoe using pybind11 and gRPC.

```
import grpc_batcher
from typing import Iterable, List, Tuple, Union


def my_func(inputs: Union[Iterable[str], Iterable[Tuple[str, int]]]) -> List[str]:
    return [({"result": i}, 0) for i in inputs]


batcher = grpc_batcher.create_batcher(
    batch_size=16,
    batch_timeout_s=.005,
    batch_process_fn=my_func,
    num_threads=16,
    track_metrics=False,
    model="repr",
)

batcher.start()
while not batcher.is_server_ready():
    print("batcher is not ready")

client = grpc_batcher.MukoeBatcherClient(server_address="localhost:50051")
print(client.send_request(b"hi"))
# => ({'result': b'hi'}, 0)
```

A few notes:
- The underlying gRPC protobuf definition is specific to dynamics and representations networks (i.e. models used in MuZero). It is designed to be easily extended.
- This is only well-tested for a Dockerfile based on ubuntu 22.04. If you are using our provided Dockerfiles, the wheel in grpc_batcher/dist/ should work well. Longer term would require integration with the manylinux toolchain which is currently not in scope.

## Building the batcher

To build grpc_batcher, we've provided a set of Dockerfiles and scripts for convenience:

```
$ ./scripts/build_grpc_batcher.sh help
Usage: ./scripts/build_grpc_batcher.sh [mode]
Modes:
  help                  Print this message.
  build_e2e             Build the entire Python wheel in a one-off Docker container.
  build_py_interactive  Build and test the Python wheel in an interactive Docker container.
  build_cpp_interactive Build the C++ components in an interactive Docker container.
  teardown              Tear down any existing Docker containers created by this script.
```

We suggest using:
- `build_e2e` if you only need to build the wheel.
- `build_cpp_interactive` for checking C++ level build correctness if modifying CPP (if there's a C++ level build issue, then `build_py_interactive` will delete the cached state which can waste cycles).
- `build_py_interactive` for building the wheel in a persistent Docker container.
- `test_py_interactive` for checking Python level correctness.
- `teardown` to teardown docker containers created in interactive modes.

