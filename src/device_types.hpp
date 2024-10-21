#pragma once

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda {
using DeviceModule = CUmodule;
using DeviceFunction = CUfunction;
using DevicePtr = CUdeviceptr;
using DeviceHandle = CUdevice;
using DeviceContext = CUcontext;
using DeviceResult = CUresult;
using DeviceEvent = cudaEvent_t;
constexpr auto DeviceSuccess = CUDA_SUCCESS;
} // namespace cuda
namespace gpu = cuda;
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace hip {
using DeviceModule = hipModule_t;
using DevicePtr = hipDeviceptr_t;
using DeviceHandle = hipDevice_t;
using DeviceContext = hipCtx_t;
using DeviceResult = hipError_t;
using DeviceFunction = hipFunction_t;
using DeviceEvent = hipEvent_t;
constexpr auto DeviceSuccess = hipSuccess;
} // namespace hip

namespace gpu = hip;
#endif
