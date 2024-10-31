#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "device_types.hpp"
#include "macro.hpp"

struct A {
  double vel, dt;
};

__device__ double *dev_ptr[2];
__device__ size_t size = 1024;

__global__ void set_dev_ptr(double *addr1, double *addr2, size_t ptr_size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid != 0)
    return;
  dev_ptr[0] = addr1;
  dev_ptr[1] = addr2;
  size = ptr_size;
}

__global__ void initialize_global_ptr_values(double value) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    dev_ptr[0][tid] = value;
    dev_ptr[1][tid] = value + 1;
  }
}

__global__ void subtract(double value) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    double tmp = dev_ptr[1][tid] - dev_ptr[0][tid];
    dev_ptr[0][tid] = tmp - 1 + value;
    dev_ptr[1][tid] = tmp - 1 + value;
  }
}

int main(int argc, const char *argv[]) {
  size_t numElements = std::atoi(argv[1]);
  double *ddata1, *ddata2;
  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;

  void *devicePtr;
  size_t my_size;
  DeviceRTErrCheck(PREFIX(GetSymbolAddress)(&devicePtr, size));
  DeviceRTErrCheck(PREFIX(Memcpy)((void *)&my_size, devicePtr, sizeof(size_t),
                                  PREFIX(MemcpyDeviceToHost)));

  std::cout << "Host Addr is " << &size << " Device PTR is " << devicePtr
            << " with value " << my_size << "\n";

  DeviceRTErrCheck(
      PREFIX(Malloc)((void **)&ddata1, numElements * sizeof(double)));
  DeviceRTErrCheck(
      PREFIX(Malloc)((void **)&ddata2, numElements * sizeof(double)));
  set_dev_ptr<<<1, 1>>>(ddata1, ddata2, numElements);
  initialize_global_ptr_values<<<num_blocks, threads>>>(10.0);
  subtract<<<num_blocks, threads>>>(10.0);
  DeviceRTErrCheck(PREFIX(DeviceSynchronize)());

  double *hdata1 = new double[numElements];
  double *hdata2 = new double[numElements];
  DeviceRTErrCheck(PREFIX(Memcpy)(hdata1, ddata1, sizeof(double) * numElements,
                                  PREFIX(MemcpyDeviceToHost)));
  DeviceRTErrCheck(PREFIX(Memcpy)(hdata2, ddata2, sizeof(double) * numElements,
                                  PREFIX(MemcpyDeviceToHost)));

  for (int i = 0; i < numElements; i++) {
    if (hdata1[i] != 10.0 || hdata2[i] != 10.0) {
      std::cout << "Values at " << i << " data1: " << hdata1[i]
                << " data2:" << hdata2[i] << " instead of 10.0 \n";
      std::cout << "Diff " << hdata1[i] - hdata2[i] << "\n";
      std::cout << "Diff " << hdata1[i] - 10.0 << "\n";
      std::cout << "Diff " << hdata2[i] - 10.0 << "\n";
      return -1;
    }
  }

  delete[] hdata1;
  delete[] hdata2;
  DeviceRTErrCheck(PREFIX(Free)(ddata1));
  DeviceRTErrCheck(PREFIX(Free)(ddata2));
  return 0;
}
