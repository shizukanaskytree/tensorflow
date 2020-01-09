#!/bin/bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package &&
# install package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
pip install /tmp/tensorflow_pkg/tf_nightly-2.0.0-cp36-cp36m-linux_x86_64.whl
