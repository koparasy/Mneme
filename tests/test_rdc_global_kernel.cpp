#if ENALBE_CUDA
#include <cuda_runtime.h>
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

__global__ void kernel_function(double *a) { *a = 123.0; }
