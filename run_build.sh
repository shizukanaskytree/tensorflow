# bazel build --config=cuda --config=dbg //tensorflow/tools/pip_package:build_pip_package
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package

# next bash run_install.sh
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg