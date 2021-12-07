# refer:
# bazel query 'deps(//tensorflow/core:__tensorflow_core_graph_graph_partition_test)' --output graph > graph.in
# grpc_tensorflow_server
# tensorflow/core/distributed_runtime/rpc/BUILD

# compose:
bazel query "deps(//tensorflow/core/distributed_runtime/rpc:grpc_tensorflow_server)" --output graph > "tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server_graph.in"

# purpose:
# 我写这个的目的就是我想自己做开发了. 但是我发现要搞的好多啊, 不可能搞得定的.
# tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server_graph.in 有 6 万行依赖关系.

# Bazel Query How-To
# https://docs.bazel.build/versions/main/query-how-to.html
