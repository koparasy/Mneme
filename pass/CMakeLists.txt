add_library(RRPass SHARED RRPass.cpp)
add_library(RR::Pass ALIAS RRPass)

target_include_directories(RRPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(RRPass PRIVATE -fno-rtti)
endif()

llvm_map_components_to_libnames(llvm_libs all)

if (ENABLE_CUDA)
target_compile_definitions(RRPass PRIVATE -DENABLE_CUDA -DENABLE_DEBUG)
elseif(ENABLE_HIP)
  target_compile_definitions(RRPass PRIVATE -DENABLE_HIP -DENABLE_DEBUG)
endif()

target_link_libraries(RRPass PRIVATE ${llvm_libs})
target_include_directories(RRPass PRIVATE "${PROJECT_SOURCE_DIR}/src")

install(TARGETS RRPass EXPORT RR 
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

