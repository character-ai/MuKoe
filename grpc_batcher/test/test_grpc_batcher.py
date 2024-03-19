# Copyright 2024 Character Technologies Inc. and Google LLC
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
import grpc_batcher
import time
from typing import List, Iterable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import msgpack
import numpy as np
from functools import partial

_NUM_CONCURRENT_REQUESTS = 10
_NUM_TOTAL_REQUESTS = 30
_MODEL_TYPE = "repr"


def my_func(inputs: Union[Iterable[str], Iterable[Tuple[str, int]]]) -> List[str]:
    time.sleep(0.01)
    if inputs:
        if isinstance(inputs[0], tuple):
            return [({"result": i[0]}, 0) for i in inputs]
        else:
            return [({"result": i}, 0) for i in inputs]


def numpy_to_bytes(arr):
    tpl = (arr.shape, arr.dtype.name, arr.tobytes("C"))
    return msgpack.packb(tpl, use_bin_type=True)


def bytes_to_numpy(data):
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    return np.frombuffer(
        buffer, dtype=np.dtype(dtype_name), count=-1, offset=0
    ).reshape(shape, order="C")


def get_inputs(inference_type: str):
    if inference_type == "dyna":
        embedding = np.random.normal(0, 1, size=(1, 6, 6, 256))
        embedding = numpy_to_bytes(embedding)
        action = 1
        return embedding, action
    else:
        observation = np.random.normal(0, 1, size=(1, 96, 96, 3, 4))
        return numpy_to_bytes(observation)


def request():
    client = grpc_batcher.MukoeBatcherClient(server_address="localhost:50051")
    data = get_inputs(_MODEL_TYPE)

    if _MODEL_TYPE == "dyna":
        embed, action = data
        send = partial(client.send_request, data=embed, action=action)
    else:
        send = partial(client.send_request, data=data)

    start = time.time()
    result = send()
    end = time.time()
    # Verify that we can convert back to numpy
    bytes_to_numpy(result[0]["result"])
    return end - start


batcher = grpc_batcher.create_batcher(
    model=_MODEL_TYPE,
    batch_size=16,
    batch_timeout_s=0.005,
    batch_process_fn=my_func,
    num_threads=16,
)

print("Starting batcher")
batcher.start()
while not batcher.is_server_ready():
    print("batcher is not ready")

print("Creating clients and sending requests...")
with ThreadPoolExecutor(max_workers=_NUM_CONCURRENT_REQUESTS) as executor:
    client = grpc_batcher.MukoeBatcherClient(server_address="localhost:50051")
    futures = [executor.submit(request) for _ in range(_NUM_TOTAL_REQUESTS)]

    # Wait for all futures to complete
    results = [future.result() for future in futures]

print("All requests processed.")
print(f"Elapsed times (average): {np.mean(results)}")
print(f"Elapsed times (median): {np.median(results)}")
batcher.print_batcher_summary()
batcher.shutdown()
