#!/bin/bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package &&
# install package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
pip uninstall --yes /tmp/tensorflow_pkg/tf_nightly-2.1.0-cp36-cp36m-linux_x86_64.whl &&
pip install /tmp/tensorflow_pkg/tf_nightly-2.1.0-cp36-cp36m-linux_x86_64.whl
