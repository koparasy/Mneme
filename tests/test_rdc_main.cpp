#include "device_types.hpp"
#include "macro.hpp"
#include <iostream>

// Declaration of the kernel from another source file
__global__ void kernel_function(double *a);

int main() {
  double *a;
  double h_a;
  DeviceRTErrCheck(PREFIX(Malloc)((void **)&a, sizeof(double)));

  // Call the function implemented in kernel.cu
  kernel_function<<<1, 1>>>(a);

  DeviceRTErrCheck(PREFIX(DeviceSynchronize)());
  DeviceRTErrCheck(PREFIX(Memcpy)((void *)&h_a, (void *)a, sizeof(double) * 1,
                                  PREFIX(MemcpyDeviceToHost)));
  DeviceRTErrCheck(PREFIX(Free)(a));

  if (h_a != 123.0) {
    std::cout << "Values differ, expected 123.0 and got " << h_a << "\n";
    return -1;
  }

  return 0;
}
