# Find Python interpreter with at least version 3.8
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter)


function(CREATE_GPU_TEST exe rdc check_source blockDims gridDims args)
  if(ENABLE_HIP)
      SET(lang HIP)
  else()
      SET(lang CUDA)
  endif()

  message(STATUS "Building test: ${exe} with the following source files:${check_source}")

  add_executable(${exe}.${lang} ${check_source} ${ARGN})

  set_source_files_properties(${check_source} PROPERTIES LANGUAGE ${lang})

  add_dependencies(${exe}.${lang} RRPass)

    # Link CUDA libraries if necessary
  if(ENABLE_CUDA)
    target_link_libraries(${exe}.${lang} PUBLIC CUDA::cudart)
    target_link_libraries(${exe}.${lang} PUBLIC ${CUDA_LIBRARIES})
  endif()
  set_source_files_properties(${check_source} PROPERTIES OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/RRPass.cpp)
  
  if (ENABLE_CUDA)
  target_compile_options(
     ${exe}.${lang}
     PUBLIC
     $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=$<TARGET_FILE:RRPass>>
   ) 
 elseif(ENABLE_HIP)
  target_compile_options(
     ${exe}.${lang}
     PUBLIC
     $<$<COMPILE_LANGUAGE:HIP>:-fpass-plugin=$<TARGET_FILE:RRPass>>
   )
 endif()

 target_include_directories(${exe}.${lang} PRIVATE "${CMAKE_SOURCE_DIR}/src/")

   if("${rdc}" STREQUAL "On")
     set_target_properties(${exe}.${lang} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
     if (ENABLE_HIP)
       target_compile_options(${exe}.${lang} PRIVATE -fgpu-rdc)
       target_link_options(${exe}.${lang} PRIVATE -fgpu-rdc)
     endif()
   endif()


   set_target_properties(${exe}.${lang} PROPERTIES CUDA_ARCHITECTURES 70)

   string(REPLACE " " "," cmd_args "${args}")

   add_test(NAME ${exe}.${lang} COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/replay_test_runner.py" "--exe" $<TARGET_FILE:${exe}.${lang}> "--record-lib" $<TARGET_FILE:record> "--replay-bin" $<TARGET_FILE:replay> "-gd" "${gridDims}" "-bd" "${blockDims}" "--args" "${cmd_args}")
endfunction()

if (ENABLE_CUDA)
message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
endif()
endif()


CREATE_GPU_TEST(test_struct "Off" test_struct.cpp "64, 128, 256, 512, 1024" "1, 2, 5, 12" "4096")
CREATE_GPU_TEST(test_redirection "Off" test_redirection.cpp "64, 128, 256, 512, 1024" "1, 2, 5, 12" "4096")
CREATE_GPU_TEST(test_free "Off" test_free.cpp "-1" "-1" "4096")
CREATE_GPU_TEST(test_device_function "Off" test_device_function.cpp "-1" "-1" "4095")

# This test tests whether we can link with an object file that contains a device function
CREATE_GPU_TEST(test_rdc_device "On" "test_rdc_device_kernel.cpp;test_rdc_device_main.cpp" "-1" "-1" "4096")

# This test tests whether we can call/link with an kernel defined on a different translation unit 
CREATE_GPU_TEST(test_rdc_global "On" "test_rdc_global_kernel.cpp;test_rdc_global_main.cpp" "-1" "-1" "4096")

if(ENABLE_HIP)
SET(lang HIP)
else()
SET(lang CUDA)
endif()
set_tests_properties("test_rdc_device.${lang}" PROPERTIES WILL_FAIL TRUE)


