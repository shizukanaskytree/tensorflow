#!/bin/bash
bazel build --config=opt \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --jobs 72 \
            --config=cuda \
            //tensorflow/tools/pip_package:build_pip_package &&
# install package
/home/wxf/tf2/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
#mv /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36d-linux_x86_64.whl /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36dm-linux_x86_64.whl &&
/home/wxf/python_debug_version/bin/pip3 uninstall --yes /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36dm-linux_x86_64.whl &&
/home/wxf/python_debug_version/bin/pip3 install /tmp/tensorflow_pkg/tf_nightly-1.13.1-cp36-cp36dm-linux_x86_64.whl

