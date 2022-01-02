#ifndef TENSORFLOW_CORE_UTIL_WRITE_LOG_H_
#define TENSORFLOW_CORE_UTIL_WRITE_LOG_H_

#include <fstream>
#include <mutex>

using std::ofstream;

namespace tensorflow {
std::mutex write_mutex;
int write_log(const std::string& input);
}
#endif  // TENSORFLOW_CORE_UTIL_WRITE_LOG_H_