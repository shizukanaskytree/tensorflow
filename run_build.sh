#!/bin/bash
bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package &&
./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
pip uninstall --yes /tmp/tensorflow_pkg/tf_nightly-2.5.0-cp36-cp36m-linux_x86_64.whl &&
pip install /tmp/tensorflow_pkg/tf_nightly-2.5.0-cp36-cp36m-linux_x86_64.whl

# --config=mkl 
