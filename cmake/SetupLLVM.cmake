# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

set(LLVM_TARGETS_TO_BUILD "all" CACHE STRING "Target architectures for LLVM")

# NOTE: When having just $LLVM_VERSION is not working out
find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG HINTS "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

message(STATUS "LLVM_FOUND: ${LLVM_FOUND}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM_LIBRARIES: ${LLVM_LIBRARIES}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
message(STATUS "LLVM_VERSION: ${LLVM_AVAILABLE_LIBS}")
set(LLVM_LIBS ${LLVM_AVAILABLE_LIBS})

#llvm_map_components_to_libnames(LLVM_LIBS support core irreader target all-targets)
#message(STATUS "LLVM MAP COMPONENTS: ${LLVM_LIBS}")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
