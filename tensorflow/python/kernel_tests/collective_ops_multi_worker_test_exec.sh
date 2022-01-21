# Test All:
# path:
# (hm) wxf@seir19:~/tf2/tensorflow$ bazel test //tensorflow/python/kernel_tests:collective_ops_multi_worker_test
# cmd:
# bazel test //tensorflow/python/kernel_tests:collective_ops_multi_worker_test

# Test a specific function
# ref post:
# How do we run a single test using Google bazel
# https://stackoverflow.com/questions/60233576/how-do-we-run-a-single-test-using-google-bazel

# path:
# (hm) wxf@seir19:~/tf2/tensorflow$
# cmd:
# bazel test //tensorflow/python/kernel_tests:collective_ops_multi_worker_test --test_filter=CollectiveOpTest.testCheckHealth
