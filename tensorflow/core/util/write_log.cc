#include <iostream>
#include <fstream>
#include <cstdlib> // for exit function
#include <string>
#include <mutex>

using std::cerr;
using std::endl;
using std::ofstream;

#include "tensorflow/core/util/write_log.h"

namespace tensorflow {

// This program output values from an array to a file named example2.dat
int write_log(const std::string& input) {
  std::lock_guard<std::mutex> write_guard(write_mutex);

  ofstream outdata;
  outdata.open("callstack_trace.log", std::ios_base::app); // opens the file
  if( !outdata ) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  outdata << input << endl;
  outdata.close();
  return 0;
}

}