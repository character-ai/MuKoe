cmake_minimum_required(VERSION 3.18)
project(GrpcBatcher VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(gRPC_BUILD_STATIC_LIBS ON CACHE BOOL "Build static gRPC libraries")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
if(Python3_FOUND)
    message(STATUS "Using Python: ${Python3_EXECUTABLE} (Version: ${Python3_VERSION})")
    message(STATUS "Python Include Dirs: ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "Python Libraries: ${PYTHON_LIBRARIES}")
else()
    message(FATAL_ERROR "Python3 Development components not found")
endif()

add_subdirectory("/grpc" ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
message(STATUS "Using gRPC via add_subdirectory.")

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_ORCA_SERVICE grpcpp_orca_service)
if(CMAKE_CROSSCOMPILING)
  find_program(_PROTOBUF_PROTOC protoc)
else()
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Pybind11
add_subdirectory("/pybind11" ${CMAKE_CURRENT_BINARY_DIR}/pybind11 EXCLUDE_FROM_ALL)
message(STATUS "Using Python: ${Python3_EXECUTABLE} (Version: ${Python3_VERSION})")
message(STATUS "Using gRPC CPP Plugin: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")

set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PROTO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_FILE "${PROTO_SOURCE_DIR}/mukoe_batcher.proto")

# Generated sources
set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/mukoe_batcher.pb.cc")
set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/mukoe_batcher.pb.h")
set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/mukoe_batcher.grpc.pb.cc")
set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/mukoe_batcher.grpc.pb.h")

# Generate protobuf and gRPC sources
add_custom_command(
    OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${PROTO_SOURCE_DIR}"
         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

include_directories(
    ${PROJECT_INCLUDE_DIR}
    ${gRPC_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Library for generated sources
add_library(mukoe_batcher_proto SHARED
  ${grpc_srcs}
  ${grpc_hdrs}
  ${proto_srcs}
  ${proto_hdrs})

target_link_libraries(mukoe_batcher_proto
    absl::base
    absl::synchronization
    absl::memory
    absl::flags
    absl::flags_parse
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})

# Main target (shared library)
add_library(grpc_batcher_lib SHARED
    "${PROJECT_SOURCE_DIR}/grpc_batcher.cc"
)

target_link_libraries(grpc_batcher_lib
  -Wl,--whole-archive
  mukoe_batcher_proto
  -Wl,--no-whole-archive
  absl::base
  absl::synchronization
  absl::memory
  absl::flags
  absl::flags_parse
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  pybind11::module
)

# Add your grpc_batcher_binding.cc to the list of sources for your library
add_library(grpc_batcher SHARED
  "${PROJECT_SOURCE_DIR}/grpc_batcher.cc"
  "${PROJECT_SOURCE_DIR}/grpc_batcher_binding.cc"
)

target_link_libraries(grpc_batcher
  -Wl,--whole-archive
  mukoe_batcher_proto
  -Wl,--no-whole-archive
  absl::base
  absl::synchronization
  absl::memory
  absl::flags
  absl::flags_parse
  grpc_batcher_lib
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  pybind11::module
)