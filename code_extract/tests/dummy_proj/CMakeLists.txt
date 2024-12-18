cmake_minimum_required(VERSION 3.28)
project("simple")

set(CMAKE_CXX_STANDARD 17)

include_directories(include/)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(SOURCE_FILES src/simple.cpp src/A.cpp)
add_executable(simple ${SOURCE_FILES})

target_link_libraries(simple m)
target_compile_options(simple PUBLIC "-fPIC")
