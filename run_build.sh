#!/bin/bash
bazel build --config=opt \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --copt=-march=native --copt=-mfpmath=both \
            --config=cuda \
            //tensorflow/tools/pip_package:build_pip_package &&
# install package
/home/wxf/tf2/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
/home/wxf/anaconda3/bin/pip3 uninstall --yes /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl &&
/home/wxf/anaconda3/bin/pip3 install /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl

