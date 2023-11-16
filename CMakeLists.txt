cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project("SVE Benchmark" VERSION 0.1)

set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "Release flags" FORCE)
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default C++ flags
if(NOT CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -Wshadow -Wno-unknown-pragmas" CACHE STRING "Default flags" FORCE)
endif()

message(STATUS "")
message(STATUS "Successfully configured ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

message(STATUS "")
