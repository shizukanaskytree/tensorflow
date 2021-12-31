// #include <boost/stacktrace.hpp>
// 
// int main()
// {
//   boost::stacktrace::safe_dump_to("./backtrace.dump");
//   return 0;
// }



#include <iostream>

#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>

void my_func_2(void) {
    std::cout << boost::stacktrace::stacktrace() << std::endl;
}

void my_func_1(double f) {
    (void)f;
    my_func_2();
}

void my_func_1(int i) {
    (void)i;
    my_func_2();
}

int main(int argc, char **argv) {
    long long unsigned int n;
    if (argc > 1) {
        n = strtoul(argv[1], NULL, 0);
    } else {
        n = 1;
    }
    for (long long unsigned int i = 0; i < n; ++i) {
        my_func_1(1);   // line 28
        my_func_1(2.0); // line 29
    }
}
