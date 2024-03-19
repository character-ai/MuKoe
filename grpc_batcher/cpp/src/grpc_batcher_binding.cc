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
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "grpc_batcher.h"

namespace py = pybind11;

std::unique_ptr<MukoeBatcher> CreateBatcher(const std::string& model, int batch_size, float batch_timeout_s, int port, int num_threads, std::function<py::object(const py::object&)>& batch_process_fn) {
    if (model == "repr") {
        return std::make_unique<MukoeReprBatcher>(batch_size, batch_timeout_s, port, num_threads, batch_process_fn);
    } else if (model == "dyna") {
        return std::make_unique<MukoeDynaBatcher>(batch_size, batch_timeout_s, port, num_threads, batch_process_fn);
    } else {
        throw std::invalid_argument("Invalid model: " + model);
    }
}

PYBIND11_MODULE(grpc_batcher, m) {
  m.doc() = "gRPC request batcher.";

  m.def("create_batcher", &CreateBatcher, "Factory function to create a batcher",
        py::arg("model"), py::arg("batch_size"), py::arg("batch_timeout_s"), py::arg("port") = 50051,
        py::arg("num_threads") = 16, py::arg("batch_process_fn"));

  py::class_<MukoeBatcher, std::unique_ptr<MukoeBatcher>>(m, "MukoeBatcher")
    .def("start", &MukoeBatcher::Start, py::call_guard<py::gil_scoped_release>())
    .def("shutdown", &MukoeBatcher::Shutdown, py::call_guard<py::gil_scoped_release>())
    .def("print_batcher_summary", &MukoeBatcher::PrintBatcherSummary, py::call_guard<py::gil_scoped_release>())
    .def("is_server_ready", &MukoeBatcher::IsServerReady, py::call_guard<py::gil_scoped_release>());

  py::class_<MukoeBatcherClient>(m, "MukoeBatcherClient", "MukoeBatcherClient sends requests to a MukoeBatcher server.")
    .def(py::init<const std::string&>(), py::arg("server_address"), "Initializes a MukoeBatcherClient with the server address.")

        // Overload for Representations net (bytes only)
        .def("send_request", [](MukoeBatcherClient& client, const std::string& data) -> py::tuple {
            py::gil_scoped_release release;  // Release the GIL before potentially blocking operation
            return client.SendRequest(data);  // This will be overloaded for "Representations"
        }, py::arg("data"), "Sends a 'Representations' request to the MukoeBatcher server and returns the result.")
        
        // Overload for Dynamics net (bytes and action)
        .def("send_request", [](MukoeBatcherClient& client, const std::string& data, int action) -> py::tuple {
            py::gil_scoped_release release;  // Release the GIL before potentially blocking operation
            return client.SendRequest(data, action);  // This will be overloaded for "Dynamics"
        }, py::arg("data"), py::arg("action"), "Sends a 'Dynamics' request to the MukoeBatcher server and returns the result.");
}
