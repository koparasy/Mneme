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
#define device_rt_call(call) CONCATENATE(DEVICE_PREFIX, call)

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
  device_rt_call(Malloc)((void **)&in, numElements * sizeof(double));
  device_rt_call(Malloc)((void **)&out, numElements * sizeof(double));
  std::cout << "In : " << in << " Out " << out << "\n";

  for (int i = 0; i < 10 ; i++){
    device_rt_call(Memset)(in, 0, numElements * sizeof(double));
    device_rt_call(Memset)(out, 0, numElements * sizeof(double));

    const int threads = 256;
    int num_blocks = (numElements + threads - 1) / threads;
    vecAdd_test<<< num_blocks, threads>>>(in, out, numElements);
    device_rt_call(DeviceSynchronize)();
  }

  double *h_in = new double[numElements];
  double *h_out = new double[numElements];
  device_rt_call(Memcpy)(h_in, in, sizeof(double)*numElements, device_rt_call(MemcpyDeviceToHost));
  device_rt_call(Memcpy)(h_out, out, sizeof(double)*numElements, device_rt_call(MemcpyDeviceToHost));
  for (int i = 0; i < numElements; i++){
    if (h_in[i] + i != h_out[i]){
      std::cout << "Values at " << i << " differ\n";
      std::cout << "Values " << h_in[i] << " " << h_out[i] << "differ\n";
      return -1;
    }
  }
  
  delete [] h_in;
  delete [] h_out;
  device_rt_call(Free)(in);
  device_rt_call(Free)(out);
  return 0;
}
