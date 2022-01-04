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

std::mutex write_mutex;

// This program output values from an array to a file named example2.dat
int write_log(const std::string& input, const std::string& f_name) {
  std::lock_guard<std::mutex> write_guard(write_mutex);
  ofstream outdata;
  outdata.open(f_name, std::ios_base::app); // opens the file
  if( !outdata ) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  outdata << input << endl;
  outdata.close();
  return 0;
}

}