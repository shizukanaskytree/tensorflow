#!/bin/bash
bazel build --config=opt \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            --jobs 72 \
            --config=cuda \
            //tensorflow/tools/pip_package:build_pip_package

