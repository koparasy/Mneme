//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"

template <typename T>
T *allocate(std::size_t size)
{
  T *ptr;
  cudaMalloc((void **)&ptr, sizeof(T) * size);
  return ptr;
}

template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

int main(int argc, char* argv[])
{
  using policy = RAJA::cuda_exec<256>;
  const std::string policy_name = "CUDA";

  std::cout << "Running vector addition with RAJA using the " << policy_name << " backend...";

  int N = std::atoi(argv[1]);
  constexpr int N_Static = 102400;


  int *a = allocate<int>(N);
  int *b = allocate<int>(N);
  int *c = allocate<int>(N);

  RAJA::forall<policy>(RAJA::TypedRangeSegment<int>(0, N_Static), [=] RAJA_HOST_DEVICE (int i) { 
    a[i] = -i;
    b[i] = i;
  });


  deallocate(a);
  deallocate(b);
  deallocate(c);
}
