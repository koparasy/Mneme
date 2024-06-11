
if (ENABLE_CUDA)
add_library(cudareplay SHARED libreplay.cpp)
target_include_directories(cudareplay PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(cudareplay PRIVATE -DENABLE_CUDA -DENABLE_DEBUG)
if (ENABLE_CXX_DEMANGLE)
    target_compile_definitions(cudareplay PRIVATE -DENABLE_CXX_DEMANGLE)
endif()
endif()
