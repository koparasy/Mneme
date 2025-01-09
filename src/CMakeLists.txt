
set(RecordLibSRC memory.cpp)
if (ENABLE_HIP)
  list(APPEND RecordLibSRC MnemeRecordHIP.cpp)
endif()

add_library(record SHARED ${RecordLibSRC})

#add_executable(replay replay.cpp memory.cpp jit.cpp common.cpp)

set(libs ${LLVM_AVAILABLE_LIBS})


target_include_directories(record SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/common)
if(ENABLE_HIP)
  target_link_libraries(record PRIVATE hip::host)
  message(STATUS "Linking with libraries: ${LLVM_AVAILABLE_LIBS}")
  #target_link_libraries(replay PRIVATE ${LLVM_AVAILABLE_LIBS})
  target_link_libraries(record PRIVATE ${LLVM_AVAILABLE_LIBS})
endif()


add_library(RR::Record ALIAS record)

#target_include_directories(replay PRIVATE ${LLVM_INCLUDE_DIRS})

#set_target_properties(replay PROPERTIES
#  INSTALL_RPATH "${LLVM_LIBRARY_DIR}")

#install(TARGETS replay)

#install(TARGETS record EXPORT RR 
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#    RUNTIME DESTINATION bin
#    INCLUDES DESTINATION include)

# Install the export set for the library
#install(EXPORT RR 
#  FILE RRTargets.cmake
#  NAMESPACE RR::
#  DESTINATION lib/cmake/RR)

# Create a version file for compatibility checks
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
#    VERSION ${PROJECT_VERSION}
#    COMPATIBILITY AnyNewerVersion)
#
## Configure the config file for the project
#configure_package_config_file(
#  "${CMAKE_SOURCE_DIR}/cmake/RRConfig.cmake.in"
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
#  INSTALL_DESTINATION lib/cmake/RR)
#
## Install the config files
#install(FILES
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfig.cmake"
#  "${CMAKE_CURRENT_BINARY_DIR}/RRConfigVersion.cmake"
#  DESTINATION lib/cmake/RR)
