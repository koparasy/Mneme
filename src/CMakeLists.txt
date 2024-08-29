

add_library(record SHARED record.cpp memory.cpp common.cpp)

set(libs ${LLVM_AVAILABLE_LIBS})


if (ENABLE_CUDA)
target_include_directories(record PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(record PRIVATE -DENABLE_CUDA -DENABLE_DEBUG)
target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
if (ENABLE_CXX_DEMANGLE)
    target_compile_definitions(record PRIVATE -DENABLE_CXX_DEMANGLE)
endif()
endif()

message(WARNING "LIBS are ${libs}")
target_link_libraries(record PRIVATE LLVM "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

add_executable(replay replay.cpp memory.cpp jit.cpp common.cpp)
target_include_directories(replay PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(replay PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(replay PRIVATE LLVM CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(replay PRIVATE -DENABLE_CUDA -DENABLE_DEBUG ${LLVM_DEFINITIONS})
target_compile_definitions(replay PRIVATE )


add_library(appreplay SHARED libreplayapp.cpp)
