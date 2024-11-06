#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define DEVICE_PREFIX hip
#elif defined(__CUDACC__)
#define DEVICE_PREFIX cuda
#include <cuda_runtime.h>
#else
#error "Cannot detect compilation type"
#endif
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define CONCATENATE_DETAIL(prefix, call) prefix##call
#define CONCATENATE(prefix, call) CONCATENATE_DETAIL(prefix, call)
#define PREFIX(call) CONCATENATE(DEVICE_PREFIX, call)

#define DeviceRTErrCheck(CALL)                                                 \
  {                                                                            \
    PREFIX(Error_t) err = CALL;                                                \
    if (err != PREFIX(Success)) {                                              \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             PREFIX(GetErrorString(err)));                                     \
      abort();                                                                 \
    }                                                                          \
  }

template <typename T> __global__ 
void vecAdd_test(T *in, T *out, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid] + tid;
  }
}

int main(int argc, const char *argv[]) {
  void *deviceAddress;
  size_t numElements = atoi(argv[1]);
  double *in, *out;
  double val = numElements;
  DeviceRTErrCheck(PREFIX(Malloc)((void **)&in, numElements * sizeof(double)));
  DeviceRTErrCheck(PREFIX(Malloc)((void **)&out, numElements * sizeof(double)));
  std::cout << "In : " << in << " Out " << out << "\n";

  for (int i = 0; i < 10 ; i++){
    DeviceRTErrCheck(PREFIX(Memset)(in, 0, numElements * sizeof(double)));
    DeviceRTErrCheck(PREFIX(Memset)(out, 0, numElements * sizeof(double)));

    const int threads = 256;
    int num_blocks = (numElements + threads - 1) / threads;
    vecAdd_test<<< num_blocks, threads>>>(in, out, numElements);
    DeviceRTErrCheck(PREFIX(DeviceSynchronize)());
  }

  double *h_in = new double[numElements];
  double *h_out = new double[numElements];
  DeviceRTErrCheck(PREFIX(Memcpy)(h_in, in, sizeof(double)*numElements, PREFIX(MemcpyDeviceToHost)));
  DeviceRTErrCheck(PREFIX(Memcpy)(h_out, out, sizeof(double)*numElements, PREFIX(MemcpyDeviceToHost)));
  for (int i = 0; i < numElements; i++){
    if (h_in[i] + i != h_out[i]){
      std::cout << "Values at " << i << " differ\n";
      std::cout << "Values " << h_in[i] << " " << h_out[i] << "differ\n";
      return -1;
    }
  }
  
  delete [] h_in;
  delete [] h_out;
  DeviceRTErrCheck(PREFIX(Free)(in));
  DeviceRTErrCheck(PREFIX(Free)(out));
  return 0;
}
