cmake_minimum_required(VERSION 3.8)

project(raja-example  LANGUAGES C CXX)

find_package(RAJA REQUIRED HINTS "${RAJA_Dir}")
find_package(RR REQUIRED HINTS "${RR_Dir}")
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

add_executable(vecAdd-raja vec_add_raja.cu)
set_target_properties(vecAdd-raja PROPERTIES CUDA_ARCHITECTURES 70)

# Get the full path of the LLVM pass plugin (MyLLVMPass)
get_target_property(RRPassPath RR::RRPass LOCATION)
message(STATUS "Record Replay Pass path: ${RRPassPath}")

target_compile_options(
 vecAdd-raja 
      PUBLIC
      $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=${RRPassPath}>
    ) 

#set_source_files_properties(vec_add_raja.cu PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
target_link_libraries(vecAdd-raja PRIVATE RAJA CUDA::cudart)
