cmake_minimum_required(VERSION 3.18)

project(RecordReplay
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_TESTS "Enable Testing" OFF)

set(LLVM_VERSION "17" CACHE STRING "LLVM version")


if(ENABLE_CUDA)
  add_definitions("-DENABLE_CUDA")
  find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_DEBUG)
  add_definitions("-DENABLE_DEBUG")
endif()

include(cmake/SetupLLVM.cmake)

add_subdirectory(pass)
add_subdirectory(src)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
