#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "device_types.hpp"
#include "macro.hpp"

struct A {
  float vel, dt, offset;
};

__device__ float compute(struct A val) { return val.offset + val.vel * val.dt; }

__global__ void test_struct_argument(float *data, struct A val,
                                     std::size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    data[tid] += (compute(val));
  }
}

int main(int argc, const char *argv[]) {
  size_t numElements = std::atoi(argv[1]);
  float *data;
  DeviceRTErrCheck(PREFIX(Malloc)((void **)&data, numElements * sizeof(float)));

  DeviceRTErrCheck(PREFIX(Memset)(data, 0, numElements * sizeof(float)));

  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;
  struct A values = {1.0, 2.0, 4.0};
  test_struct_argument<<<num_blocks, threads>>>(data, values, numElements);
  DeviceRTErrCheck(PREFIX(DeviceSynchronize)());

  float *h_data = new float[numElements];
  DeviceRTErrCheck(PREFIX(Memcpy)(h_data, data, sizeof(float) * numElements,
                                  PREFIX(MemcpyDeviceToHost)));

  for (int i = 0; i < numElements; i++) {
    if (h_data[i] != values.vel * values.dt + values.offset) {
      std::cout << "Values at " << i << " " << h_data[i]
                << " instate of 2.0 \n";
      return -1;
    }
  }

  delete[] h_data;
  DeviceRTErrCheck(PREFIX(Free)(data));
  return 0;
}
