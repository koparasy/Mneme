#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#include <assert.h>
#include <iostream>

#include "memory.hpp"

namespace memory {
namespace cuda {
template <typename Ty> Ty *advanceVoidPtr(Ty *Ptr, int64_t Offset) {
  static_assert(std::is_void<Ty>::value);
  return const_cast<char *>(reinterpret_cast<const char *>(Ptr) + Offset);
}

#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    auto err = CALL;                                                           \
    if (err != CUDA_SUCCESS) {                                                 \
      printf("ERROR @ %s:%d ->  %d\n", __FILE__, __LINE__, err);               \
      abort();                                                                 \
    }                                                                          \
  }

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

void MemoryUnMap(void *DevPtr, CUmemGenericAllocationHandle &MemHandle,
                 size_t MappedMemSize) {
  CUdeviceptr DVAddr = reinterpret_cast<CUdeviceptr>(DevPtr);

  cuErrCheck(cuMemUnmap(DVAddr, MappedMemSize));

  cuErrCheck(cuMemRelease(MemHandle));

  cuErrCheck(cuMemAddressFree(DVAddr, MappedMemSize));
}

CUmemGenericAllocationHandle MemMapToDevice(void **DevPtr, void *req_addr,
                                            uint64_t MemSize,
                                            uint64_t &CeiledSize,
                                            int device_id) {
  CUmemAllocationProp Prop = {};
  size_t Granularity = 0;

  // TODO: Check whether nvidia supports only pinned device types
  Prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  Prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // I am not sure whether id means DeviceID
  // TODO: Currently only using the device id 0. We need to revisit this.
  Prop.location.id = device_id;
  cuErrCheck(cuMemGetAllocationGranularity(
      &Granularity, &Prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  if (Granularity == 0) {
    throw std::runtime_error("Unsopported Granularity");
  }

  // Ceil to page size.
  CeiledSize = (MemSize + Granularity - 1) / Granularity * Granularity;

  // Create a handler of our allocation
  CUmemGenericAllocationHandle AHandle;
  cuErrCheck(cuMemCreate(&AHandle, CeiledSize, &Prop, 0));

  CUdeviceptr devPtr = 0;
  cuErrCheck(cuMemAddressReserve(&devPtr, CeiledSize, Granularity,
                                 reinterpret_cast<CUdeviceptr>(req_addr), 0));

  cuErrCheck(cuMemMap(devPtr, CeiledSize, 0, AHandle, 0));

  CUmemAccessDesc ADesc = {};
  ADesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ADesc.location.id = 0;
  ADesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Sets address
  cuErrCheck(cuMemSetAccess(devPtr, CeiledSize, &ADesc, 1));

  *DevPtr = (void *)(devPtr);
  std::cout << "Device Address " << *DevPtr << "\n";
  return AHandle;
}
} // namespace cuda
} // namespace memory
