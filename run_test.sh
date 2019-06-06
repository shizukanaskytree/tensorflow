#!/bin/bash
set -x
#TARGET="//tensorflow/core/distributed_runtime/rpc:grpc_session_test"
#TARGET="//tensorflow/core/distributed_runtime/rpc:grpc_util_test"
#TARGET="//tensorflow/core/distributed_runtime/rpc:grpc_tensor_coding_test"
#bazel test --config=opt \
#           --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
#           --config=cuda \
#           $TARGET

TARGET="//tensorflow/core/distributed_runtime/rpc:rpc_tests"
bazel run --config=opt \
           --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
           --config=cuda \
           $TARGET
