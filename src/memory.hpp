#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace memory {
namespace cuda {
void MemoryUnMap(void *DevPtr, CUmemGenericAllocationHandle &MemHandle,
                 size_t MappedMemSize);
CUmemGenericAllocationHandle MemMapToDevice(void **DevPtr, void *req_addr,
                                            uint64_t MemSize,
                                            uint64_t &CeiledSize,
                                            int device_id = 0);

} // namespace cuda
} // namespace memory
