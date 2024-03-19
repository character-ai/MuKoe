/** Copyright 2024 The MuKoe Authors
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==============================================================================*/
#ifndef GRPC_BATCHER_H_
#define GRPC_BATCHER_H_
#include "mukoe_batcher.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/embed.h>
#include <algorithm>
#include <numeric>

using grpc::Server;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerReaderWriter;
using grpc::Channel;
using mukoe_batcher::MukoeRequest;
using mukoe_batcher::MukoeResponse;
using mukoe_batcher::ModelResponse;
using mukoe_batcher::KeyValue;
using mukoe_batcher::MukoeService;

namespace py = pybind11;

struct QueueItem {
  MukoeRequest request;
  std::promise<MukoeResponse> promise;
};

class ThreadSafeQueue {
public:
  void push(QueueItem&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(std::move(value));  // Use emplace to construct the object in place
    cond_var_.notify_one();
  }

  bool pop(QueueItem& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void wait_and_pop(QueueItem& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this] { return !queue_.empty(); });
    value = std::move(queue_.front());
    queue_.pop();
  }

  bool pop_nowait(QueueItem& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return false; // No items available, return immediately
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

private:
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::queue<QueueItem> queue_;
};

struct BatchProcessingMetrics {
  long long batch_size_sum = 0;
  long long end_to_end_time_sum = 0;
  long long batch_build_time_sum = 0;
  long long py_conversion_time_sum = 0;
  long long process_time_sum = 0;
  long long cpp_conversion_time_sum = 0;
  long long postprocess_time_sum = 0;

  int count = 0;

  void add_data_point(long long batch_size, long long end_to_end_time, long long batch_build_time,
                      long long py_conversion_time, long long process_time, 
                      long long cpp_conversion_time, long long postprocess_time) {
    batch_size_sum += batch_size;
    end_to_end_time_sum += end_to_end_time;
    batch_build_time_sum += batch_build_time;
    py_conversion_time_sum += py_conversion_time;
    process_time_sum += process_time;
    cpp_conversion_time_sum += cpp_conversion_time;
    postprocess_time_sum += postprocess_time;
    count++;
  }

  double get_average_batch_size() const { return count ? static_cast<double>(batch_size_sum) / count : 0; }
  double get_average_end_to_end_time() const { return count ? static_cast<double>(end_to_end_time_sum) / count : 0; }
  double get_average_batch_build_time() const { return count ? static_cast<double>(batch_build_time_sum) / count : 0; }
  double get_average_py_conversion_time() const { return count ? static_cast<double>(py_conversion_time_sum) / count : 0; }
  double get_average_process_time() const { return count ? static_cast<double>(process_time_sum) / count : 0; }
  double get_average_cpp_conversion_time() const { return count ? static_cast<double>(cpp_conversion_time_sum) / count : 0; }
  double get_average_postprocess_time() const { return count ? static_cast<double>(postprocess_time_sum) / count : 0; }

  void print_summary() const {
    std::cout << "MukoeBatcher statistics:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    std::cout << "Total data points: " << count << std::endl;
    std::cout << "Average batch size: " << get_average_batch_size() << std::endl;
    std::cout << "Average end to end time (ms): " << get_average_end_to_end_time() << std::endl;
    std::cout << "Average batch build time (ms): " << get_average_batch_build_time() << std::endl;
    std::cout << "Average Python conversion time (ms): " << get_average_py_conversion_time() << std::endl;
    std::cout << "Average process time (ms): " << get_average_process_time() << std::endl;
    std::cout << "Average C++ conversion time (ms): " << get_average_cpp_conversion_time() << std::endl;
    std::cout << "Average postprocess time (ms): " << get_average_postprocess_time() << std::endl;
  }
};

/**
* @class MukoeServiceImpl 
* @brief Implements a mukoe_batcher gRPC service.
*
* MukoeServiceImpl is responsible for collecting individual gRPC requests,
* and routing them to MukoeBatcher, which then batches requests together,
* runs some type of processing (typically a Python function) and returns
* the results back to the client.
*/
class MukoeServiceImpl final : public MukoeService::Service {
public:
  /**
  * Constructs a MukoeServiceImpl with the given parameters:
  * @param request_queue: The ThreadSafeQueue where individual requests are placed.
  */
  MukoeServiceImpl(ThreadSafeQueue& request_queue) : request_queue_(request_queue) {}

  /**
  * The logic for handling a gRPC request.
  */
  Status ProcessRequest(ServerContext* context, ServerReaderWriter<MukoeResponse, MukoeRequest>* stream);

private:
  ThreadSafeQueue& request_queue_;
};

/**
* @class MukoeBatcher
* @brief Implements a MukoeBatcher.
*
* MukoeBatcher is a class that's responsible for spinning up a gRPC server
* that accepts individual requests (in bytes) and batches them together for
* processing. The processing function is defined by the user, but is intended
* to be a Python function that calls an ML Framework like JAX or PyTorch.
*/
class MukoeBatcher {
public:
  /**
  * Constructs a MukoeBatcher.
  * 
  * @param batch_size The maximum batch size to process.
  * @param batch_timeout_s The maximum amount of time (in seconds) to wait before processing a batch.
  * @param port The port on which the gRPC server should listen.
  * @param num_threads The number of gRPC server threads to spin up.
  * @param batch_process_fn The function to call for processing each batch of requests.
  */
  MukoeBatcher(
    int batch_size, float batch_timeout_s, int port, int num_threads,
    std::function<py::object(const py::object&)> batch_process_fn);

  virtual ~MukoeBatcher() = default;
  
  /**
  * Starts the gRPC server and begins handling requests.
  */
  void Start();
  
  /**
  * Shuts down the gRPC server.
  */
  void Shutdown();

  /**
  * Checks if the gRPC server is ready to serve requests.
  *
  * @return True if the server is ready, false otherwise.
  */
  bool IsServerReady();

  /**
  * Prints timing summaries of various operations within BuildAndProcessBatches.
  */
  void PrintBatcherSummary() const {
    batch_processing_metrics_.print_summary();
  }

private:
  void RunServer();
  void BuildAndProcessBatches();
  virtual py::list ConvertToPyList_(const std::vector<MukoeRequest>& batchData) = 0;
  uint batch_size_;
  float batch_timeout_s_;
  int port_;
  int num_threads_;
  std::function<py::object(const py::object&)> batch_process_fn_;
  std::unique_ptr<Server> server_;
  std::thread server_thread_;
  std::thread batcher_thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool is_serving_;
  ThreadSafeQueue request_queue_;
  BatchProcessingMetrics batch_processing_metrics_;
};

class MukoeReprBatcher : public MukoeBatcher {
public:
  using MukoeBatcher::MukoeBatcher;  // Inherit constructor

private:
  py::list ConvertToPyList_(const std::vector<MukoeRequest>& batchData) override;
};


class MukoeDynaBatcher : public MukoeBatcher {
public:
  using MukoeBatcher::MukoeBatcher;  // Inherit constructor

private:
  py::list ConvertToPyList_(const std::vector<MukoeRequest>& batchData) override;
};


/**
 * @class MukoeBatcherClient Impl
 * @brief Client Implementation for sending requests to the MukoeBatcher server.
 * 
 * MukoeBatcherClientImpl is used to connect to a MukoeBatcher server and send requests to it.
 * It encapsulates the details of setting up a gRPC channel and making RPC calls.
 */
class MukoeBatcherClientImpl {
public:
  /**
  * Constructs a GrpcBatcherClientImpl connected to the specified server.
  * 
  * @param channel The gRPC channel to use for communication with the server.
  */
  MukoeBatcherClientImpl(std::shared_ptr<Channel> channel) : stub_(MukoeService::NewStub(channel)) {}

  /**
  * Sends a representations net request to the GrpcBatcher server.
  * 
  * @param data The data to send as part of the request.
  * @return Data from MukoeBatcher.
  */
  MukoeResponse SendRequest(const std::string& data);

  /**
  * Sends a dynamics net request to the GrpcBatcher server.
  * 
  * @param data The data to send as part of the request.
  * @param action The associated action
  * @return Data from MukoeBatcher.
  */
  MukoeResponse SendRequest(const std::string& data, int action);


private:
  std::unique_ptr<MukoeService::Stub> stub_;
  std::vector<long long> processTimes_;
  MukoeResponse SendRequest_(MukoeRequest& request);
};

/**
* @class MukoeBatcherClient
* @brief Wrapper Client for MukoeBatcherClientImpl.
*
* MukoeBatcherClient exposes a smaller amount of functionality for
* downstream consumption.
*/
class MukoeBatcherClient {
public:

  /**
  * Constructs a MukoeBatcherClient.
  *
  * @param server_address The address of the MukoeBatcher server.
  */
  MukoeBatcherClient(const std::string& server_address);

  /**
  * Sends a representations net request to the MukoeBatcherClient.
  *
  * @param data The data to send as part of the request.
  * @return Data from MukoeBatcher.
  */
  py::tuple SendRequest(const std::string& data);

  /**
  * Sends a dynamics net request to the MukoeBatcherClient.
  *
  * @param data The data to send as part of the request.
  * @param action The associated action.
  * @return Data from MukoeBatcher.
  */
  py::tuple SendRequest(const std::string& data, int action);

private:
  std::unique_ptr<MukoeBatcherClientImpl> client_;
  py::tuple ResponseToPython_(const MukoeResponse& response);
};

#endif
