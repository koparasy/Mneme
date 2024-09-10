cmake_minimum_required(VERSION 3.8)

project(cuda-example  LANGUAGES C CXX CUDA)

find_package(RR REQUIRED HINTS "${RR_Dir}")
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)


add_executable(vecAdd-cuda vec_add.cu)
set_target_properties(vecAdd-cuda PROPERTIES CUDA_ARCHITECTURES 70)

# Get the full path of the LLVM pass plugin (MyLLVMPass)
get_target_property(RRPassPath RR::RRPass LOCATION)
message(STATUS "Record Replay Pass path: ${RRPassPath}")

target_compile_options(
 vecAdd-cuda 
      PUBLIC
      $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=${RRPassPath}>
    ) 



target_link_libraries(vecAdd-cuda PRIVATE CUDA::cudart)
