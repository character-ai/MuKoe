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
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import sysconfig


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Note - specifying grpc_install_dir here specifically
        # to enable the build through the Docker route. But this is
        # probably not the right scalable solution.
        grpc_install_dir = os.environ.get("GRPC_INSTALL_DIR", "/grpc")
        python_include_dir = sysconfig.get_path("include")
        python_library = sysconfig.get_config_var("LIBDIR")

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPython_EXECUTABLE=" + sys.executable,
            "-DPython_INCLUDE_DIR=" + python_include_dir,
            "-DPython_LIBRARY=" + python_library,
            "-DCMAKE_PREFIX_PATH=" + grpc_install_dir,
            "-DgRPC_BUILD_TESTS=OFF",
            f"-DProtobuf_DIR={grpc_install_dir}/lib/cmake/protobuf",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose", "--parallel", "16"] + build_args,
            cwd=self.build_temp,
        )

        # Rename the output from libgrpc_batcher.so to grpc_batcher.so
        os.rename(
            os.path.join(extdir, "libgrpc_batcher.so"),
            os.path.join(extdir, "grpc_batcher.so"),
        )


setup(
    name="grpc_batcher",
    version="0.0.1",
    author="Allen Wang & Wendy Shang",
    author_email="allencwang@google.com, wendy@character.ai",
    description="A simple gRPC request batcher",
    long_description="A gRPC request batcher implemented in C++ commonly used for RL and ML serving applications.",
    ext_modules=[CMakeExtension("grpc_batcher", "cpp")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
