

add_library(record SHARED record.cpp memory.cpp common.cpp)
add_executable(replay replay.cpp memory.cpp jit.cpp common.cpp)

set(libs ${LLVM_AVAILABLE_LIBS})


target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
if (ENABLE_CUDA)
target_include_directories(record PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(record PRIVATE -DENABLE_CUDA -DENABLE_DEBUG)
target_link_libraries(record PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(replay PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(replay PRIVATE LLVM CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(replay PRIVATE -DENABLE_CUDA -DENABLE_DEBUG ${LLVM_DEFINITIONS})
target_link_libraries(record PRIVATE LLVM "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

if (ENABLE_CXX_DEMANGLE)
    target_compile_definitions(record PRIVATE -DENABLE_CXX_DEMANGLE)
endif()
elseif(ENABLE_HIP)
  target_compile_definitions(replay PRIVATE __HIP_PLATFORM_AMD__)
  target_compile_definitions(record PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(replay PRIVATE hip::host)
  target_include_directories(replay PRIVATE ${HIP_INCLUDE_DIRS})
  target_include_directories(record PRIVATE ${HIP_INCLUDE_DIRS})
  message(STATUS "Linking with libraries: ${LLVM_AVAILABLE_LIBS}")
  target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})
  target_link_libraries(record PRIVATE ${LLVM_AVAILABLE_LIBS})
endif()


add_library(RR::Record ALIAS record)

target_include_directories(replay PRIVATE ${LLVM_INCLUDE_DIRS})

set_target_properties(replay PROPERTIES
  INSTALL_RPATH "${LLVM_LIBRARY_DIR}")

install(TARGETS replay)

install(TARGETS record EXPORT RR 
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

# Install the export set for the library
install(EXPORT RR 
  FILE RRTargets.cmake
  NAMESPACE RR::
  DESTINATION lib/cmake/RR)

# Create a version file for compatibility checks
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

# Configure the config file for the project
configure_package_config_file(
  "${CMAKE_SOURCE_DIR}/cmake/RRConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
  INSTALL_DESTINATION lib/cmake/RR)

# Install the config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
  DESTINATION lib/cmake/RR)
