#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "DeviceTraits.hpp"
#include "MnemeMemoryBlob.hpp"
#include "Logger.hpp"
#include "Utils.hpp"

namespace mneme {

class MnemeMemoryBlobHIP;
template <> struct DeviceTraits<MnemeRecorderHIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using AllocGranularityFlags = hipMemAllocationGranularity_flags;
};

class MnemeMemoryBlobHIP : public MnemeMemoryBlob<MnemeMemoryBlobHIP> {
private:
  static inline uint64_t
  getPageSize(int DeviceID,
              const hipMemAllocationGranularity_flags Granularity) {
    uint64_t PageSize;
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    // TODO: I could not find any documentation regarding the compressionType in
    // HIP. I will leave unitialized a.t.m.
    // Prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    hipErrCheck(hipMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
    DBG(Logger::logs("mneme") << "Page Size is : " << PageSize << "\n";)
    return PageSize;
  }

public:
  void allocate(hipMemGenericAllocationHandle_t &MHandle, uintptr_t Addr,
                uintptr_t Size, int DeviceId) {
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceId;

    hipErrCheck(hipMemCreate(&MHandle, Size, &Prop, 0));
    hipErrCheck(hipMemMap((void *)Addr, Size, 0, MHandle, 0));

    hipMemAccessDesc ADesc = {};
    ADesc.location.type = hipMemLocationTypeDevice;
    ADesc.location.id = DeviceId;
    ADesc.flags = hipMemAccessFlagsProtReadWrite;

    // Sets address
    DBG(Logger::logs("mneme") << "Setting Access 'RW' to " << (void *)Addr
                              << " with size " << Size << "\n");
    hipErrCheck(hipMemSetAccess((void *)Addr, Size, &ADesc, 1));
  }

  static uint64_t getMinPageSize(int DeviceID) {
    return getPageSize(DeviceID, hipMemAllocationGranularityMinimum);
  }

  static void *getVirtualAddress(uint64_t Size, uintptr_t VA,
                                 uint64_t Alignment) {
    hipDeviceptr_t devPtr = 0;

    hipErrCheck(hipMemAddressReserve(&devPtr, Size, Alignment,
                                     reinterpret_cast<hipDeviceptr_t>(VA), 0));
    DBG(Logger::logs("mneme")
        << "Allocated VASize "
        << (double)((double)Size / (1024L * 1024L * 1024L)) << " at Address "
        << std::hex << devPtr << std::dec << "\n");
    return (void *)devPtr;
  }

  void release(hipMemGenericAllocationHandle_t &MHandle, uintptr_t Addr,
               uintptr_t Size) {
    hipErrCheck(hipMemUnmap((void *)Addr, Size));
    hipErrCheck(hipMemRelease(MHandle));
    hipErrCheck(hipMemAddressFree((void *)Addr, Size));
  }
};

} // namespace mneme
