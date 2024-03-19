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
#include "grpc_batcher.h"

using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;


Status MukoeServiceImpl::ProcessRequest(ServerContext* context, ServerReaderWriter<MukoeResponse, MukoeRequest>* stream) {
  while (true) {
    QueueItem item;
    if (stream->Read(&(item.request))) {
      try {
        auto future = item.promise.get_future();
        request_queue_.push(std::move(item));

        MukoeResponse response = future.get();

        if (!stream->Write(response)) {
          // Exit the loop if we can't write to the stream
          break;
        }
      } catch (const std::exception& e) {
        std::cerr << "ProcessRequest::Exception caught: " << e.what() << std::endl;
        break;
      }
    } else {
      // Exit the loop if there's no more data to read, i.e. the client has finished sending requests.
      break;
    }
  }
  return Status::OK;
}


MukoeBatcher::MukoeBatcher(int batch_size, float batch_timeout_s, int port, int num_threads, std::function<py::object(const py::object&)> batch_process_fn)
  : batch_size_(batch_size), batch_timeout_s_(batch_timeout_s), port_(port), num_threads_(num_threads), batch_process_fn_(batch_process_fn) {}


bool MukoeBatcher::IsServerReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_serving_;
}


void MukoeBatcher::RunServer() {
  std::string server_address = "0.0.0.0:" + std::to_string(port_);

  MukoeServiceImpl service(request_queue_);

  ServerBuilder builder;
  //builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_GZIP);
  builder.SetMaxReceiveMessageSize(1 * 1024 * 1024); // Max receive size 1MB
  builder.SetMaxSendMessageSize(1 * 1024 * 1024); // Max send size 1MB
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Set the number of threads in the server's thread pool for handling incoming RPCs
  builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, num_threads_);
  builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, num_threads_);
  builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, num_threads_);

  server_ = builder.BuildAndStart();

  // Signal that the server is ready
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_serving_ = true;
  }
  cv_.notify_one();

  // Start the server and block until the server is shut down
  server_->Wait();
}

void MukoeBatcher::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_serving_ = false;  // Ensure to set is_serving_ to false to stop the loop in BuildAndProcessBatches
  }
  cv_.notify_all();  // Notify all waiting threads

  if(server_) {
    server_->Shutdown();
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
  }
  if (batcher_thread_.joinable()) {
    batcher_thread_.join();
  }
}

void MukoeBatcher::BuildAndProcessBatches() {
  // Wait until the server is up and running
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return is_serving_; });
  lock.unlock();
  std::chrono::steady_clock::time_point batch_start_time = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point batch_built_time;
  std::chrono::steady_clock::time_point py_converted_time;
  std::chrono::steady_clock::time_point processed_time;
  std::chrono::steady_clock::time_point cpp_converted_time;
  std::chrono::steady_clock::time_point postprocessed_time;

  while (is_serving_) {
    std::vector<QueueItem> queueItems;
    std::vector<MukoeRequest> batchData;

    // Attempt to build a batch of strings
    QueueItem item;
    while (batchData.size() < batch_size_) {
      if (request_queue_.pop_nowait(item)) {
        batchData.push_back(item.request);
        queueItems.push_back(std::move(item));
      } else {
        // If no item was popped, check if the batch timeout has been reached
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - batch_start_time);
        if (elapsed.count() >= static_cast<int>(batch_timeout_s_ * 1000)) {
          break; // Batch timeout reached, proceed to process the batch
        }
        std::this_thread::yield(); // Yield to allow other threads to run
      }
    }
    batch_built_time = std::chrono::steady_clock::now();

    if (!batchData.empty()) {
      // Batch has been built!
      std::vector<MukoeResponse> results;
      try {
        // Convert MukoeRequests to a Python compatible format.
        py::gil_scoped_acquire acquire;
        py::list batchDataPyList = ConvertToPyList_(batchData);
        py_converted_time = std::chrono::steady_clock::now();
        py::object resultsPyList = batch_process_fn_(batchDataPyList);
        processed_time = std::chrono::steady_clock::now();

        // Converts results from a list of py::tuples back to a vector
        // of MukoeResponses.
        for (py::handle result : resultsPyList) {
          MukoeResponse response;

          py::tuple resultTuple = result.cast<py::tuple>();
          py::dict modelResponseDict = resultTuple[0].cast<py::dict>();
          int step = resultTuple[1].cast<int>();

          ModelResponse* modelResponse = response.mutable_result();

          for (auto item : modelResponseDict) {
            // Extract key and value from the item, which is a key-value pair in the dictionary
            std::string key = py::str(item.first);
            py::handle value = item.second;

            KeyValue* keyValue = modelResponse->add_dictionary();
            keyValue->set_key(key);
            keyValue->set_values(value.cast<py::bytes>().cast<std::string>());
          }
          modelResponse->set_step(step);
          results.push_back(response);
        }
        cpp_converted_time = std::chrono::steady_clock::now();
      } catch (...) {
        // If an exception is caught, set the same exception for all promises
        auto exception_ptr = std::current_exception();
        for (auto& item : queueItems) {
          item.promise.set_exception(exception_ptr);
        }
        continue;  // Skip to the next iteration
      }

      // Ensure the number of results matches the number of input strings
      if (results.size() != queueItems.size()) {
        // Handle error: number of results does not match the number of input strings
        // For simplicity, we set an error for all items in this batch
        for (auto& item : queueItems) {
          item.promise.set_exception(std::make_exception_ptr(std::runtime_error("Mismatch between input and output batch sizes")));
        }
      } else {
        // Resolve each promise with the corresponding result
        for (size_t i = 0; i < queueItems.size(); ++i) {
          queueItems[i].promise.set_value(results[i]);
        }
        postprocessed_time = std::chrono::steady_clock::now();
      }
      batch_processing_metrics_.add_data_point(
          static_cast<long long>(batchData.size()),  // batch size
          std::chrono::duration_cast<std::chrono::milliseconds>(postprocessed_time - batch_start_time).count(),  // end to end time
          std::chrono::duration_cast<std::chrono::milliseconds>(batch_built_time - batch_start_time).count(),  // batch build time
          std::chrono::duration_cast<std::chrono::milliseconds>(py_converted_time - batch_built_time).count(),  // Python conversion time
          std::chrono::duration_cast<std::chrono::milliseconds>(processed_time - py_converted_time).count(),  // process time
          std::chrono::duration_cast<std::chrono::milliseconds>(cpp_converted_time - processed_time).count(),  // C++ conversion time
          std::chrono::duration_cast<std::chrono::milliseconds>(postprocessed_time - cpp_converted_time).count()  // postprocess time
      );
      batch_start_time = std::chrono::steady_clock::now();
    }
  }
}


void MukoeBatcher::Start() {
    // Starts the server and batch processing threads
    server_thread_ = std::thread([this]() { RunServer(); });
    batcher_thread_ = std::thread([this]() { BuildAndProcessBatches(); });
}


py::list MukoeReprBatcher::ConvertToPyList_(const std::vector<MukoeRequest>& batchData) {
  py::list batchDataPyList;
  for (const auto& request : batchData) {
    // Assuming MukoeRequest has a member function representations() returning the data
    batchDataPyList.append(py::bytes(request.representations().data()));
  }
  return batchDataPyList;
}


py::list MukoeDynaBatcher::ConvertToPyList_(const std::vector<MukoeRequest>& batchData) {
  py::list batchDataPyList;
  for (const auto& request : batchData) {
    // Assuming MukoeRequest has a member function dynamics() returning the data and action
    py::tuple dynamicsItem = py::make_tuple(py::bytes(request.dynamics().data()), py::int_(request.dynamics().action()));
    batchDataPyList.append(dynamicsItem);
  }
  return batchDataPyList;
}


MukoeResponse MukoeBatcherClientImpl::SendRequest_(MukoeRequest& request) {
  ClientContext context;
  MukoeResponse response;
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  auto stream = stub_->ProcessRequest(&context);
  if (!stream->Write(request)) {
    std::cerr << "Failed to write request to stream." << std::endl;
  }
  stream->WritesDone();
  while (stream->Read(&response)) {}

  // Check the status of the RPC after finishing the reads
  Status status = stream->Finish();
  if (!status.ok()) {
    std::cerr << "RPC failed. Error: " << status.error_message() << std::endl;
  }
  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
  processTimes_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
  return response;
}


// Overload for "Representations" (bytes only)
MukoeResponse MukoeBatcherClientImpl::SendRequest(const std::string& data) {
    MukoeRequest request;
    request.mutable_representations()->set_data(data);  // Using the 'representations' field
    return SendRequest_(request);
}


// Overload for "Dynamics" (bytes and action)
MukoeResponse MukoeBatcherClientImpl::SendRequest(const std::string& data, int action) {
    MukoeRequest request;
    auto dynamics = request.mutable_dynamics();  // Using the 'dynamics' field
    dynamics->set_data(data);
    dynamics->set_action(action);
    return SendRequest_(request);
}


MukoeBatcherClient::MukoeBatcherClient(const std::string& server_address) {
  grpc::ChannelArguments channelArgs;
  channelArgs.SetMaxReceiveMessageSize(1 * 1024 * 1024); // Max receive size of 1MB
  channelArgs.SetMaxSendMessageSize(1 * 1024 * 1024); // Max send size of 1MB
  auto channel = CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), channelArgs);
  client_ = std::make_unique<MukoeBatcherClientImpl>(channel);
}


py::tuple MukoeBatcherClient::ResponseToPython_(const MukoeResponse& response) {
  py::gil_scoped_acquire acquire;
  py::dict dict;
  for (const auto& kv : response.result().dictionary()) {
    dict[py::str(kv.key())] = py::bytes(kv.values());
  }
  // Create the tuple with the dictionary and the step
  py::tuple result = py::make_tuple(dict, response.result().step());
  return result;
}


py::tuple MukoeBatcherClient::SendRequest(const std::string& data) {
  return ResponseToPython_(client_->SendRequest(data));
}


py::tuple MukoeBatcherClient::SendRequest(const std::string& data, int action) {
  return ResponseToPython_(client_->SendRequest(data, action));
}
