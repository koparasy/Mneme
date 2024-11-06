//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"
#include <stdexcept>
#ifdef ENABLE_CUDA
#define PREFIX(x) cuda##x
#include <cuda_runtime.h>
using policy = RAJA::cuda_exec_occ_max<RAJA::named_usage::unspecified>;
#elif defined(ENABLE_HIP)
#define PREFIX(x) hip##x
#include <hip/hip_runtime.h>
using policy = RAJA::hip_exec_occ_max<RAJA::named_usage::unspecified>;
#endif

#define DeviceRTErrCheck(CALL)                                                 \
  {                                                                            \
    PREFIX(Error_t) err = CALL;                                                \
    if (err != PREFIX(Success)) {                                              \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             PREFIX(GetErrorString(err)));                                     \
      abort();                                                                 \
    }                                                                          \
  }

template<typename T>
void copy(T* device, T* host, int size){
  DeviceRTErrCheck(PREFIX(Memcpy)(host, device, sizeof(T)*size, PREFIX(MemcpyDeviceToHost)));
}

template <typename T>
T *allocate(std::size_t size)
{
  T *ptr;
  DeviceRTErrCheck(PREFIX(Malloc)((void **)&ptr, sizeof(T) * size));
  return ptr;
}

template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
    DeviceRTErrCheck(PREFIX(Free)(ptr));
    ptr = nullptr;
  }
}

int main(int argc, char* argv[])
{
  int N = std::atoi(argv[1]);


  int *a = allocate<int>(N);
  int *b = allocate<int>(N);
  int *c = allocate<int>(N);
  int *c_h = new int[N];

  RAJA::forall<policy>(RAJA::TypedRangeSegment<int>(0, N), [=] RAJA_HOST_DEVICE (int i) { 
    a[i] = -i;
    b[i] = i;
  });

  RAJA::forall<policy>(RAJA::TypedRangeSegment<int>(0, N), [=] RAJA_HOST_DEVICE (int i) { 
    c[i] = a[i] + b[i];
  });

  copy(c, c_h, N);
  deallocate(a);
  deallocate(b);
  deallocate(c);

  for (int i = 0; i < N ; i++){
    if (c_h[i] != 0.0) {
      std::cout << "Incorrect result at index " << i << " expected the value of 0.0 but got " << c_h[i] << "\n";
    }
  }


  delete [] c_h;
}
