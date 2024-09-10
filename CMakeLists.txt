cmake_minimum_required(VERSION 3.18)

project(RR
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
if(BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_C_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CUDA_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG_INIT}")
  add_compile_options("-gdwarf-4")
elseif(BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_C_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CUDA_FLAGS_DEBUG_INIT "-gdwarf-4" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG_INIT}")
  message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG_INIT}")
  add_compile_options("-gdwarf-4")
endif()

option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_TESTS "Enable Testing" OFF)

set(LLVM_VERSION "17" CACHE STRING "LLVM version")

include(CMakePackageConfigHelpers)

if(ENABLE_CUDA)
  enable_language(CUDA)
  add_definitions("-DENABLE_CUDA")
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER "clang++")
  message(STATUS "Cuda Libraries: ${CUDA_LIBRARIES}")
  message(STATUS "Cuda Library: ${CUDA_LIBRARY}")
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
