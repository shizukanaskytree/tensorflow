#include <iostream>
#include <fstream> // std::ifstream

#include <boost/stacktrace.hpp>
#include <boost/filesystem.hpp>

int main(int argc, char **argv) {

  std::string fullName;
  std::cout << "Type your full path of backtrace.dump: ";
  std::getline (std::cin, fullName);
  std::cout << "Your path is: " << fullName;

  if (boost::filesystem::exists(fullName)) {
      // std::ifstream ifs("./backtrace.dump");
      std::ifstream ifs(fullName);

      boost::stacktrace::stacktrace st = boost::stacktrace::stacktrace::from_dump(ifs);
      std::cout << "backtrace.dump:\n" << st << std::endl;

      // cleaning up
      ifs.close();
      // boost::filesystem::remove("./backtrace.dump");

      std::cout << std::endl;
  } else {
    std::cout << "Type your full path of backtrace.dump: ";
  }
}
