#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

struct A{
  int k;
  double b, c[7];
};

template <typename T> __global__ 
void vecAdd(T *in, T *out, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid];
  }
}

int main(int argc, const char *argv[]) {
  size_t numElements = atoi(argv[1]);
  float *in, *out;
  cudaMalloc((void **)&in, numElements * sizeof(float));
  cudaMalloc((void **)&out, numElements * sizeof(float));
  std::cout << "In : " << in << " Out " << out << "\n";
  A tmp;
  tmp.k = 42;
  tmp.b = 123;

  cudaMemset(in, 1, numElements * sizeof(float));
  cudaMemset(out, 0, numElements * sizeof(float));

  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;
  vecAdd<<< num_blocks, threads>>>(in, out, numElements);
  cudaDeviceSynchronize();

  float *h_in = new float[numElements];
  float *h_out = new float[numElements];
  cudaMemcpy(h_in, in, sizeof(float)*numElements, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out, out, sizeof(float)*numElements, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numElements; i++){
    if (h_in[i] != h_out[i]){
      std::cout << "Values at " << i << " differ\n";
      return -1;
    }
  }
  
  delete [] h_in;
  delete [] h_out;
  cudaFree(in);
  cudaFree(out);
  return 0;
}
