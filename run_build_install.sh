bazel build --config=cuda --config=dbg //tensorflow/tools/pip_package:build_pip_package
# bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall -y /tmp/tensorflow_pkg/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
pip install /tmp/tensorflow_pkg/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
