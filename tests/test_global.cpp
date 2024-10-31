#include <cuda_runtime.h>
__device__ double value = 0.0;

__global__ void test_global() { value = 1.0; }

int main(int argc, const char *argv[]) { test_global<<<1, 1>>>(); }
