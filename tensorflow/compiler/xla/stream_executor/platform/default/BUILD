load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.bzl", "tsl_copts")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow/compiler/xla/stream_executor:__subpackages__",
    "//tensorflow/stream_executor:__subpackages__",
])

cc_library(
    name = "platform",
    textual_hdrs = ["initialize.h"],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tsl_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/tsl/platform:dso_loader",
        "@com_google_absl//absl/strings",
    ],
)
