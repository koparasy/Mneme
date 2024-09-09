#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

struct A{
  double x,y;
};

__device__ A DINOS_DINOS = { 1.0, 1.0 };

template <typename T> __global__ 
void vecAdd(T *in, T *out, T val, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid] + val + DINOS_DINOS.x;
  }
}

int main(int argc, const char *argv[]) {
  size_t numElements = atoi(argv[1]);
  double *in, *out;
  double val = numElements;
  cudaMalloc((void **)&in, numElements * sizeof(double));
  cudaMalloc((void **)&out, numElements * sizeof(double));
  std::cout << "In : " << in << " Out " << out << "\n";

  cudaMemset(in, 0, numElements * sizeof(double));
  cudaMemset(out, 0, numElements * sizeof(double));

  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;
  vecAdd<<< num_blocks, threads>>>(in, out, val, numElements);
  cudaDeviceSynchronize();

  double *h_in = new double[numElements];
  double *h_out = new double[numElements];
  cudaMemcpy(h_in, in, sizeof(double)*numElements, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out, out, sizeof(double)*numElements, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numElements; i++){
    if (h_in[i] + 1.0 + val != h_out[i]){
      std::cout << "Values at " << i << " differ\n";
      std::cout << "Values " << h_in[i] << " " << h_out[i] << "differ\n";
      return -1;
    }
  }
  
  delete [] h_in;
  delete [] h_out;
  cudaFree(in);
  cudaFree(out);
  return 0;
}
