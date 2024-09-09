#include <cuda_runtime.h>
#include <iostream>

// Declaration of the kernel from another source file
__global__ void kernel_function(double *a);

int main() {
  double *a;
  double h_a;
  cudaMalloc((void **)&a, sizeof(double));

  // Call the function implemented in kernel.cu
  kernel_function<<<1, 1>>>(a);

  cudaDeviceSynchronize();
  cudaMemcpy((void *)&h_a, (void *)a, sizeof(double) * 1,
             cudaMemcpyDeviceToHost);
  cudaFree(a);

  if (h_a != 123.0) {
    std::cout << "Values differ, expected 123.0 and got " << h_a << "\n";
    return -1;
  }

  return 0;
}
