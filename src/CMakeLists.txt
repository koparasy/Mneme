

add_library(record SHARED record.cpp memory.cpp common.cpp)

set(libs ${LLVM_AVAILABLE_LIBS})


if (ENABLE_CUDA)
target_include_directories(record PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(record PRIVATE -DENABLE_CUDA -DENABLE_DEBUG)
target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(record PRIVATE CUDA::cudart CUDA::cuda_driver)
if (ENABLE_CXX_DEMANGLE)
    target_compile_definitions(record PRIVATE -DENABLE_CXX_DEMANGLE)
endif()
endif()

target_link_libraries(record PRIVATE LLVM "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
add_library(RR::Record ALIAS record)

add_executable(replay replay.cpp memory.cpp jit.cpp common.cpp)
target_include_directories(replay PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(replay PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(replay PRIVATE LLVM CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(replay PRIVATE -DENABLE_CUDA -DENABLE_DEBUG ${LLVM_DEFINITIONS})
target_compile_definitions(replay PRIVATE )

set_target_properties(replay PROPERTIES
  INSTALL_RPATH "${LLVM_LIBRARY_DIR}"
    # If you want to use RPATH during build time as well, enable the following
  BUILD_WITH_INSTALL_RPATH TRUE
)

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
