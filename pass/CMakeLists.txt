add_library(RRPass SHARED RRPass.cpp)

target_include_directories(RRPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(RRPass PRIVATE -fno-rtti)
endif()

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(RRPass
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

install(TARGETS RRPass)
