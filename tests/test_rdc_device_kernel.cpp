#if ENALBE_CUDA
#include <cuda_runtime.h>
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif
#include <iostream>

__device__ void kernel_function_device(double *a) { *a = 123.0; }
