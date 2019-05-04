#!/bin/bash
bazel build --config=opt \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    --config=cuda \
    //tensorflow/tools/pip_package:build_pip_package &&
/home/wxf/tf2/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
pip uninstall --yes /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl &&
pip install /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl &&
pip uninstall --yes tf-estimator-nightly &&
pip install tf-estimator-nightly==1.14.0.dev2019031401
