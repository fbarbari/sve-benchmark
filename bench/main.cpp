#include <iostream>
#include <benchmark/benchmark.h>

#if defined(__GNUG__)
// g++
#define COMPILER_NAME "GCC"
#define COMPILER_VERSION __VERSION__
#elif defined()
// clang++
#define COMPILER_NAME "clang"
#define COMPILER_VERSION __VERSION__
#else
// Unknown/unsupported compiler
#define COMPILER_NAME "unknown"
#define COMPILER_VERSION "0.0.0"
#endif

int main(int argc, char** argv) {
    std::cout << "Compiled with " << COMPILER_NAME << " version " << COMPILER_VERSION << " on "
              << __DATE__ << " " << __TIME__ << std::endl;

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return 0;
}
