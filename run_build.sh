#!/bin/bash
bazel build --config=opt --config=nonccl \
            --verbose_failures \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --cxxopt="-flax-vector-conversions" \
            //tensorflow/tools/pip_package:build_pip_package &&
# install package
bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag ./.. &&
pip3 uninstall --yes ./../tf_nightly-1.13.1-cp36-cp36m-linux_aarch64.whl &&
pip3 install ./../tf_nightly-1.13.1-cp36-cp36m-linux_aarch64.whl

