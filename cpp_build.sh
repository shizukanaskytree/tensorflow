#!/bin/bash
bazel build --config=opt \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --config=cuda \
            //tensorflow/tools/lib_package:libtensorflow
