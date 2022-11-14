# docs: https://www.tensorflow.org/install/source
./configure
bazelisk build --config=cuda //tensorflow/tools/pip_package:build_pip_package
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg

### Install the package
# pip install /tmp/tensorflow_pkg/tensorflow-