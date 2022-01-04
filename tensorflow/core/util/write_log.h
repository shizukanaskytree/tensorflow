#ifndef TENSORFLOW_CORE_UTIL_WRITE_LOG_H_
#define TENSORFLOW_CORE_UTIL_WRITE_LOG_H_

#include <fstream>
#include <mutex>
using std::ofstream;

// 使用:
//
// std::string FILE_NAME = "/home/wxf/tf2/tensorflow/tensorflow/core/distributed_runtime/debug.log";
// write_log(check_health_status.ToString(), FILE_NAME);

// option = 0 : default
//

namespace tensorflow {
int write_log(const std::string& input, const std::string& f_name="/home/wxf/tf2/tensorflow/callstack_trace.log");
}
#endif  // TENSORFLOW_CORE_UTIL_WRITE_LOG_H_