load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "xla-translate",
    testonly = True,
    srcs = ["xla_translate_main.cc"],
    deps = [
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_transfer_manager",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:translate_registration",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:translate_registration",
        "//tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla:translate_registration",
        "//tensorflow/tsl/platform:platform_port",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
