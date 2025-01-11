#pragma once

#include <cstdint>
#include <iostream>

#include "DeviceTraits.hpp"
#include "Logger.hpp"
#include "Utils.hpp"

namespace mneme {
template <typename ImplT> class MnemeMemoryBlob {
public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;
  using MemoryAllocationHandle =
      typename DeviceTraits<ImplT>::MemoryAllocationHandle;

private:
  uint64_t ActualSize;

protected:
  MemoryAllocationHandle MemHandle;
  uintptr_t BlobAddr;
  uint64_t Size;
  uintptr_t DeviceID;

public:
  MnemeMemoryBlob(uintptr_t Addr, uintptr_t Size, int DeviceID = 0)
      : Size(Size), DeviceID(DeviceID) {
    auto MinPageSize = ImplT::getMinPageSize(DeviceID);
    ActualSize = util::roundUp(Size, MinPageSize);
    void *VA = static_cast<ImplT>(*this).getVirtualAddress(ActualSize, Addr,
                                                           MinPageSize);
    DBG(Logger::logs("mneme") << "Requested Addr: " << std::hex << Addr
                              << " Reserved Addr: " << VA << "\n");

    static_cast<ImplT &>(*this).allocate(MemHandle, BlobAddr, Size, DeviceID);
    BlobAddr = reinterpret_cast<uintptr_t>(VA);
  };

  void release() {
    if (!BlobAddr)
      return;
    static_cast<ImplT &>(*this).release(MemHandle, BlobAddr, ActualSize);
    BlobAddr = 0;
  }

  MnemeMemoryBlob(const MnemeMemoryBlob &) = delete;
  MnemeMemoryBlob &operator=(const MnemeMemoryBlob &) = delete;

  MnemeMemoryBlob &operator=(MnemeMemoryBlob &&other) noexcept {
    if (this != &other) {
      MemHandle = other.MHandle;
      BlobAddr = other.BlobAddr;
      Size = other.Size;
      ActualSize = other.ActualSize;
      DeviceID = other.DeviceID;
      other.BlobAddr = 0;
    }
    return *this;
  }

  MnemeMemoryBlob(MnemeMemoryBlob &&other) noexcept
      : MemHandle(other.MemHandle), BlobAddr(other.BlobAddr), Size(other.Size),
        DeviceID(other.DeviceID) {
    other.BlobAddr = 0;
  }
};
}
