#ifndef TENSORFLOW_CORE_UTIL_WRITE_LOG_H_
#define TENSORFLOW_CORE_UTIL_WRITE_LOG_H_

#include <fstream>
#include <mutex>
using std::ofstream;

// 使用:
//
// std::string FILE_NAME = "/home/wxf/tf2/tensorflow/tensorflow/core/distributed_runtime/debug.log";
// write_log(check_health_status.ToString(), FILE_NAME);

// boost::stacktrace::to_string(boost::stacktrace::stacktrace())
// //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");

namespace tensorflow {
int write_log(int pid, const char* func, int line, const char* file,
  const std::string& save_f_name="/home/wxf/tf2/tensorflow/cc_callstack_trace.log",
  const std::string& callstack="");
}
#endif  // TENSORFLOW_CORE_UTIL_WRITE_LOG_H_