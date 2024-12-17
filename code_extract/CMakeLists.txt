cmake_minimum_required(VERSION 3.31)
project("code_extractor")
set(TOOL_NAME "code-extract")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

link_directories(${LLVM_LIBRARY_DIRS})

set(SOURCE src/Visitor.cpp src/code_extract.cpp src/ToolManager.cpp src/VisitManager.cpp)
add_executable(${TOOL_NAME} ${SOURCE})

target_include_directories(${TOOL_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_compile_options(${TOOL_NAME} PUBLIC "-fPIC")
target_link_libraries(${TOOL_NAME}
    LLVMSupport
    clangTooling
    clangBasic
    clangASTMatchers
    clangAST
    clangFrontend
)