cmake_minimum_required(VERSION 3.8)

project(raja-example  LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_HIP "Enable CUDA" OFF)

find_package(RAJA REQUIRED HINTS "${RAJA_Dir}")
find_package(RR REQUIRED HINTS "${RR_Dir}")

if(ENABLE_CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
add_definitions("-DENABLE_CUDA")
set(DEVICE_LANGUAGE "CUDA")
elseif(ENABLE_HIP)
  enable_language(HIP)
  find_package(HIP REQUIRED)
  add_definitions("-DENABLE_HIP")
set(DEVICE_LANGUAGE "HIP")
else()
  message(FATAL_ERROR "Please define either '-DENABLE_HIP' or -DENABLE_CUDA")
endif()


add_executable(vecAdd-raja vec_add_raja.cu)

# Get the full path of the LLVM pass plugin (MyLLVMPass)
get_target_property(RRPassPath RR::RRPass LOCATION)
message(STATUS "Record Replay Pass path: ${RRPassPath}")
message(STATUS "DEVICE LANGUAGE is: ${DEVICE_LANGUAGE}")

set_source_files_properties(vec_add_raja.cu PROPERTIES LANGUAGE ${DEVICE_LANGUAGE})

target_compile_options(
 vecAdd-raja 
      PUBLIC
      $<$<COMPILE_LANGUAGE:${DEVICE_LANGUAGE}>:-fpass-plugin=${RRPassPath}>
    ) 

#set_source_files_properties(vec_add_raja.cu PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
if (ENABLE_CUDA)
target_link_libraries(vecAdd-raja PRIVATE RAJA CUDA::cudart)
elseif(ENABLE_HIP)
  target_include_directories(vecAdd-raja PRIVATE ${HIP_INCLUDE_DIRS})
target_link_libraries(vecAdd-raja PRIVATE RAJA hip::host)
endif()
