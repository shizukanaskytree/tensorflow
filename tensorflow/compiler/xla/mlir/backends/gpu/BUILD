load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow/compiler/xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "xla-gpu-opt",
    srcs = ["xla-gpu-opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/mlir/backends/gpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
