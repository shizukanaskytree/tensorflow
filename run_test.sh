bazel run //tensorflow/core/distributed_runtime:collective_rma_distributed_test



### All tests (for C++ changes).
# $ bazel test //tensorflow/...

### All Python tests (for Python front-end changes).
# $ bazel test //tensorflow/python/...

### All tests (with GPU support).
# $ bazel test -c opt --config=cuda //tensorflow/...
# $ bazel test -c opt --config=cuda //tensorflow/python/...


# bazel run //tensorflow/python/kernel_tests:string_split_op_test

# bazel run //tensorflow/python:special_math_ops_test

### Or you can go to the individual directory and run all the tests there
# cd python/kernel_tests
# bazel run :one_hot_op_test


# Example run:

# (hm) wxf@seir19:~/tf2/tensorflow/tensorflow/python/eager$ bazel run :remote_test
# Starting local Bazel server and connecting to it...
# INFO: Options provided by the client:
#   Inherited 'common' options: --isatty=1 --terminal_columns=130
# INFO: Reading rc options for 'run' from /home/wxf/tf2/tensorflow/.bazelrc:
#   Inherited 'common' options: --experimental_repo_remote_exec
# INFO: Reading rc options for 'run' from /home/wxf/tf2/tensorflow/.bazelrc:
#   Inherited 'build' options: --define framework_shared_object=true --java_toolchain=@tf_toolchains//toolchains/java:tf_java_toolchain --host_java_toolchain=@tf_toolchains//toolchains/java:tf_java_toolchain --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --enable_platform_specific_config --define=with_xla_support=true --config=short_logs --config=v2 --define=no_aws_support=true --define=no_hdfs_support=true
# INFO: Reading rc options for 'run' from /home/wxf/tf2/tensorflow/.tf_configure.bazelrc:
#   Inherited 'build' options: --action_env PYTHON_BIN_PATH=/home/wxf/anaconda3/envs/hm/bin/python3 --action_env PYTHON_LIB_PATH=/home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages --python_path=/home/wxf/anaconda3/envs/hm/bin/python3 --action_env CUDA_TOOLKIT_PATH=/usr/local/cuda-11.1 --action_env TF_CUDA_COMPUTE_CAPABILITIES=3.5,7.0 --action_env LD_LIBRARY_PATH=:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 --action_env GCC_HOST_COMPILER_PATH=/usr/bin/x86_64-linux-gnu-gcc-7 --config=cuda
# INFO: Reading rc options for 'run' from /home/wxf/tf2/tensorflow/.bazelrc:
#   Inherited 'build' options: --deleted_packages=tensorflow/compiler/mlir/tfrt,tensorflow/compiler/mlir/tfrt/benchmarks,tensorflow/compiler/mlir/tfrt/jit/python_binding,tensorflow/compiler/mlir/tfrt/jit/transforms,tensorflow/compiler/mlir/tfrt/python_tests,tensorflow/compiler/mlir/tfrt/tests,tensorflow/compiler/mlir/tfrt/tests/saved_model,tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu,tensorflow/core/runtime_fallback,tensorflow/core/runtime_fallback/conversion,tensorflow/core/runtime_fallback/kernel,tensorflow/core/runtime_fallback/opdefs,tensorflow/core/runtime_fallback/runtime,tensorflow/core/runtime_fallback/util,tensorflow/core/tfrt/common,tensorflow/core/tfrt/eager,tensorflow/core/tfrt/eager/backends/cpu,tensorflow/core/tfrt/eager/backends/gpu,tensorflow/core/tfrt/eager/core_runtime,tensorflow/core/tfrt/eager/cpp_tests/core_runtime,tensorflow/core/tfrt/fallback,tensorflow/core/tfrt/gpu,tensorflow/core/tfrt/run_handler_thread_pool,tensorflow/core/tfrt/runtime,tensorflow/core/tfrt/saved_model,tensorflow/core/tfrt/saved_model/tests,tensorflow/core/tfrt/tpu,tensorflow/core/tfrt/utils
# INFO: Found applicable config definition build:short_logs in file /home/wxf/tf2/tensorflow/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
# INFO: Found applicable config definition build:v2 in file /home/wxf/tf2/tensorflow/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
# INFO: Found applicable config definition build:cuda in file /home/wxf/tf2/tensorflow/.bazelrc: --repo_env TF_NEED_CUDA=1 --crosstool_top=@local_config_cuda//crosstool:toolchain --@local_config_cuda//:enable_cuda
# INFO: Found applicable config definition build:linux in file /home/wxf/tf2/tensorflow/.bazelrc: --copt=-w --host_copt=-w --define=PREFIX=/usr --define=LIBDIR=$(PREFIX)/lib --define=INCLUDEDIR=$(PREFIX)/include --define=PROTOBUF_INCLUDE_PATH=$(PREFIX)/include --cxxopt=-std=c++14 --host_cxxopt=-std=c++14 --config=dynamic_kernels --distinct_host_configuration=false --experimental_guard_against_concurrent_changes
# INFO: Found applicable config definition build:dynamic_kernels in file /home/wxf/tf2/tensorflow/.bazelrc: --define=dynamic_loaded_kernels=true --copt=-DAUTOLOAD_DYNAMIC_KERNELS
# INFO: Analyzed target //tensorflow/python/eager:remote_test (414 packages loaded, 29090 targets configured).
# INFO: Found 1 target...
# Target //tensorflow/python/eager:remote_test up-to-date:
#   bazel-bin/tensorflow/python/eager/remote_test
# INFO: Elapsed time: 1539.142s, Critical Path: 516.93s
# INFO: 18922 processes: 3032 internal, 15890 local.
# INFO: Build completed successfully, 18922 total actions
# INFO: Build completed successfully, 18922 total actions


# Case 2:
# Go to see:
# tensorflow/python/kernel_tests/collective_ops_multi_worker_test_exec.sh