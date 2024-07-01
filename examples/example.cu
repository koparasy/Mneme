#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>

struct A{
  int k;
  double b, c[7];
};

__device__ A value;

template <typename T> __global__ 
void vecAdd(T *in, T *out, size_t size, A tmp, int8_t c) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size)
    return;
  auto stride = gridDim.x * blockDim.x;

  for (; tid < size; tid += stride) {
    out[tid] += in[tid]  + tmp.k + tmp.b + c + value.b;
  if (tid == 0 ){
    printf("%f %d %g\n", out[tid], tmp.k, tmp.b);
  }
  }


}

int main(int argc, const char *argv[]) {
  size_t numElements = atoi(argv[1]);
  float *in, *out;
  cudaMalloc((void **)&in, numElements * sizeof(float));
  cudaMalloc((void **)&out, numElements * sizeof(float));
  A tmp;
  tmp.k = 42;
  tmp.b = 123;

  cudaMemset(in, 0, numElements * sizeof(float));
  cudaMemset(out, 0, numElements * sizeof(float));

  const int threads = 256;
  int num_blocks = (numElements + threads - 1) / threads;
  printf("Before call\n");
  vecAdd<<< num_blocks, threads>>>(in, out, numElements, tmp, 10);


  cudaFree(in);
  cudaFree(out);
  return 0;
}
