#!/bin/bash
bazel build --config=opt \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --copt=-march=native --copt=-mfpmath=both \
            //tensorflow/tools/pip_package:build_pip_package &&
#bazel build --config=opt \
#            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
#            --copt=-march=native --copt=-mfpmath=both \
#            --config=mkl \
#            --config=cuda \
#            //tensorflow/tools/pip_package:build_pip_package &&
# install package
bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg &&
pip uninstall --yes /tmp/tensorflow_pkg/tf_nightly-2.1.0-cp36-cp36m-linux_x86_64.whl &&
pip install /tmp/tensorflow_pkg/tf_nightly-2.1.0-cp36-cp36m-linux_x86_64.whl
