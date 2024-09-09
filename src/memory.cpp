#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include <iostream>
#include <sys/types.h>

#include "macro.hpp"
#include "memory.hpp"

#ifdef ENABLE_CUDA
namespace cuda {

uint64_t getPageSize(int device_id,
                     const CUmemAllocationGranularity_flags Granularity) {
  uint64_t PageSize;
  CUmemAllocationProp Prop = {};
  Prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  Prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  Prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

  cuErrCheck(cuMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
  return PageSize;
}

uint64_t getReccommendedPageSize(int device_id) {
  return getPageSize(device_id, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
}

uint64_t getMinPageSize(int device_id) {
  return getPageSize(device_id, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

MemoryBlob::MemoryBlob(uintptr_t Addr, uintptr_t sz, int device_id)
    : BlobAddr(Addr), Size(sz), DeviceID((device_id)) {
  CUmemAllocationProp Prop = {};
  Prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  Prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // I am not sure whether id means DeviceID
  // TODO: Currently only using the device id 0. We need to revisit this.
  Prop.location.id = device_id;

  cuErrCheck(cuMemCreate(&AHandle, Size, &Prop, 0));
  cuErrCheck(cuMemMap(Addr, Size, 0, AHandle, 0));

  CUmemAccessDesc ADesc = {};
  ADesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ADesc.location.id = 0;
  ADesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Sets address
  cuErrCheck(cuMemSetAccess(Addr, Size, &ADesc, 1));
  return;
}

MemoryBlob::MemoryBlob(MemoryBlob &&other) noexcept
    : AHandle(other.AHandle), BlobAddr(other.BlobAddr), Size(other.Size),
      DeviceID(other.DeviceID) {}

void *ReserveVirtualAddress(void *req_addr, uint64_t VASize,
                            uint64_t PageSize) {
  CUdeviceptr devPtr = 0;
  std::cout << "Requesting VASize "
            << (double)((double)VASize / (1024L * 1024L * 1024L))
            << " at Address " << std::hex << req_addr << "\n";
  cuErrCheck(cuMemAddressReserve(&devPtr, VASize, PageSize,
                                 reinterpret_cast<CUdeviceptr>(req_addr), 0));
  return (void *)devPtr;
}

MemoryBlob &MemoryBlob::operator=(MemoryBlob &&other) noexcept {
  if (this != &other) {
    AHandle = other.AHandle;
    BlobAddr = other.BlobAddr;
    Size = other.Size;
    DeviceID = other.DeviceID;
  }
  return *this;
}

void MemoryBlob::release() {
  cuErrCheck(cuMemUnmap(BlobAddr, Size));
  cuErrCheck(cuMemRelease(AHandle));
}

}; // namespace cuda
#endif

ContiguousAddrBlock::ContiguousAddrBlock(uintptr_t start, uint64_t sz)
    : PageAddr(start), Size(sz) {}

// Comparison operators for sorting blocks by address and size
bool ContiguousAddrBlock::operator<(const ContiguousAddrBlock &other) const {
  return PageAddr < other.PageAddr;
}

// Erase a block from the multiset
bool PageManager::EraseVirtualAddress(uintptr_t addr, size_t size) {
  auto range = FreeVARanges.equal_range(ContiguousAddrBlock{addr, size});
  for (auto it = range.first; it != range.second; ++it)
    if (it->Size == size) {
      FreeVARanges.erase(it); // Erase the block and return success
      return true;
    }
  return false; // ContiguousAddrBlock not found
}

// Function to coalesce contiguous blocks
void PageManager::coalesce() {
  if (FreeVARanges.size() < 2)
    return; // Nothing to coalesce if less than two blocks

  auto it = FreeVARanges.begin();
  while (it != FreeVARanges.end()) {
    auto next_it = std::next(it);

    if (next_it == FreeVARanges.end())
      break;

    // Check if the current block is contiguous with the next block
    if (it->PageAddr + it->Size == next_it->PageAddr) {
      // Merge the two blocks
      ContiguousAddrBlock mergedBlock = {it->PageAddr,
                                         it->Size + next_it->Size};

      // Remove the original two blocks
      it = FreeVARanges.erase(it);
      next_it = FreeVARanges.erase(next_it);

      // Insert the merged block
      FreeVARanges.insert(mergedBlock);

      // Start over from the merged block (to ensure we handle multiple
      // contiguous blocks)
      it = FreeVARanges.find(mergedBlock);
    } else {
      ++it;
    }
  }
}

// Find a block that has a size >= requestedSize
std::multiset<ContiguousAddrBlock>::iterator
PageManager::findFreeBlock(size_t requestedSize) {
  for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
    auto size = it->Size;
    if (size >= requestedSize) {
      return it;
    }
  }
  return FreeVARanges.end();
}

// Find a block that includes the range [Addr, Addr + size)

std::multiset<ContiguousAddrBlock>::iterator
PageManager::findInclusivePage(uintptr_t Addr, size_t size) {
  uintptr_t request_end = Addr + size;
  for (auto it = FreeVARanges.begin(); it != FreeVARanges.end(); ++it) {
    uintptr_t block_end = it->PageAddr + it->Size;
    if (it->PageAddr <= Addr && block_end >= request_end) {
      return it;
    }
  }
  return FreeVARanges.end();
}

std::pair<uintptr_t, uint64_t>
PageManager::ReserveBestFitPage(uint64_t VASize) {
  // We need to always reserve at least a single page
  uint64_t ReqSize = util::RoundUp(VASize, PageSize);
  auto FreeNode = findFreeBlock(ReqSize);
  if (FreeNode == FreeVARanges.end())
    return std::make_pair((uintptr_t) nullptr, 0);

  auto Ptr = FreeNode->PageAddr;
  auto NodePageSize = FreeNode->Size;

  FreeVARanges.erase(FreeNode);

  if (ReqSize == NodePageSize)
    return std::make_pair(Ptr, ReqSize);

  auto NewPtr = Ptr + ReqSize;
  auto RemainingSize = NodePageSize - ReqSize;

  ContiguousAddrBlock block{NewPtr, RemainingSize};
  FreeVARanges.insert(block);

  // This can be expensive. Currently we coalesce in every request that
  // modifies our free-pages.
  coalesce();

  return std::make_pair(Ptr, ReqSize);
}

std::pair<uintptr_t, uint64_t> PageManager::RequestExactPage(uint64_t VASize,
                                                             void *VA) {
  // We need to always reserve at least a single page
  std::cout << "Requesting exact page\n";
  uint64_t ReqSize = util::RoundUp(VASize, PageSize);
  auto FreeNode = findInclusivePage((uintptr_t)VA, ReqSize);
  if (FreeNode == FreeVARanges.end())
    return std::make_pair((uintptr_t) nullptr, 0);

  auto Ptr = FreeNode->PageAddr;
  auto NodePageSize = FreeNode->Size;

  std::cout << "Returned start: " << std::hex << Ptr << " End: " << std::hex
            << Ptr + NodePageSize << "\n";

  FreeVARanges.erase(FreeNode);

  // We found exactly the requested page.
  if (ReqSize == NodePageSize && (uintptr_t)VA == Ptr)
    return std::make_pair(Ptr, ReqSize);

  if (VA != nullptr) {
    if ((uintptr_t)VA < Ptr ||
        ((uintptr_t)VA + VASize) > (Ptr + NodePageSize)) {
      std::ostringstream oss;
      oss << "Unable to return requested address: " << std::hex
          << reinterpret_cast<uintptr_t>(VA)
          << " instead the returned address is " << std::hex
          << reinterpret_cast<uintptr_t>(Ptr) << "\n";
      throw std::runtime_error(oss.str());
    }
  }

  // There are 'unused' addresses left from the requested one
  // We add them back to the page manager
  auto NewNodePageSize = (uintptr_t)VA - Ptr;
  ContiguousAddrBlock block{Ptr, NewNodePageSize};
  FreeVARanges.insert(block);

  // There are 'unused' addresses right/higher than the end of
  // the requested page addresses
  auto NewPtr = (uintptr_t)VA + ReqSize;
  if (NewPtr < Ptr + NodePageSize) {
    ContiguousAddrBlock block{NewPtr, Ptr + NodePageSize - NewPtr};
    FreeVARanges.insert(block);
  }
  // This can be expensive. Currently we coalesce in every request that
  // modifies our free-pages.
  coalesce();

  return std::make_pair((uintptr_t)VA, ReqSize);
}

std::pair<uintptr_t, uint64_t> PageManager::AllocatePage(uint64_t VASize,
                                                         void *VA) {
  if (VA == nullptr)
    return ReserveBestFitPage(VASize);
  return RequestExactPage(VASize, VA);
}

PageManager::PageManager(uint64_t VASize, void *VA, int32_t device_id) {
  DeviceID = device_id;
  PageSize = gpu::getMinPageSize(DeviceID);
  TotalVASize = util::RoundUp(VASize, PageSize);
  ReservedVA = (uintptr_t)gpu::ReserveVirtualAddress(VA, TotalVASize, PageSize);

  std::cout << "Reserved Address Ranges : " << std::hex << ReservedVA
            << " of size " << ((double)TotalVASize) / (1024 * 1024) << "MB"
            << " " << TotalVASize << " PageSize: " << PageSize / 1024.0 << "\n";
  FreeVARanges.insert(ContiguousAddrBlock{ReservedVA, TotalVASize});
}

PageManager::~PageManager() {
  std::cout << "Releasing Reserved VA Memory\n";
  std::cout << "ReservedVA is " << std::hex << ReservedVA << "\n";
  std::cout << "Total Size is " << ((double)TotalVASize / (1024.0 * 1024.0))
            << " MB\n";
  cuErrCheck(cuMemAddressFree(ReservedVA, TotalVASize));
}

/**
 * @brief Allocates memory from the device and optionally maps it to the
 * requested address.
 *
 * @param VASize The size of the memory allocation.
 * @param VA The requested address.
 *
 * @return MemoryBlob A MemoryBlob of at least VASize that contains the
 * requested Addr.
 */
gpu::MemoryBlob PageManager::AllocateMemoryBlob(uint64_t VASize, void *VA) {
  auto [AllocatedPagedAddr, Size] = AllocatePage(VASize, VA);
  // Once we have the allocated address we need to Allocate the memory
  auto MemBlob = gpu::MemoryBlob(AllocatedPagedAddr, Size);
  return std::move(MemBlob);
}

void PageManager::ReleaseMemoryBlob(gpu::MemoryBlob &&Blob) {
  Blob.release();
  ContiguousAddrBlock AddrBlock = {Blob.BlobAddr, Blob.Size};
  FreeVARanges.insert(AddrBlock);
  coalesce();
}

MemoryManager::MemoryManager(uint64_t VASize, void *VA, int32_t device_id)
    : FreePages(VASize, VA, device_id) {}

void *MemoryManager::allocate(uint64_t size, void *req_addr) {
  gpu::MemoryBlob Blob = FreePages.AllocateMemoryBlob(size, req_addr);
  void *Addr = (void *)Blob.BlobAddr;
  if (req_addr != nullptr && Addr != req_addr) {
    if (Addr < req_addr ||
        (uintptr_t)req_addr + size > (uintptr_t)Addr + Blob.Size) {
      std::ostringstream oss;
      oss << "Requested Memory Addr" << std::hex
          << reinterpret_cast<uintptr_t>(req_addr) << "With size: " << size
          << " memory manager returned address [Low:" << std::hex
          << Blob.BlobAddr << ", High:" << std::hex << Blob.BlobAddr + Blob.Size
          << "]\n";
      throw std::runtime_error(oss.str());
    }

    AllocatedMemory.emplace(req_addr, std::move(Blob));
    std::cout << "Returning addr " << std::hex << (uintptr_t)req_addr << "\n";
    return req_addr;
  }
  AllocatedMemory.emplace(Addr, std::move(Blob));
  std::cout << "Returning addr " << std::hex << (uintptr_t)Addr << "\n";
  return Addr;
}

void MemoryManager::release(void *addr) {
  auto Blob = AllocatedMemory.find(addr);
  if (Blob == AllocatedMemory.end()) {
    std::ostringstream oss;
    oss << "Released Memory " << std::hex << reinterpret_cast<uintptr_t>(addr)
        << " is not tracked by memory manager\n";
    throw std::runtime_error(oss.str());
  }
  Blob->second.release();
  AllocatedMemory.erase(Blob);
}

uintptr_t MemoryManager::StartVAAddr() const { return FreePages.ReservedVA; }
uint64_t MemoryManager::TotalVASize() const { return FreePages.TotalVASize; }

llvm::raw_fd_ostream &operator<<(llvm::raw_fd_ostream &os,
                                 const MemoryManager &MemManager) {
  // Serialize the object to the llvm::raw_ostream
  uint64_t maxSize = 0;
  for (auto &KV : MemManager.AllocatedMemory) {
    if (maxSize < KV.second.Size)
      maxSize = KV.second.Size;
  }

  if (maxSize == 0)
    return os;

  uint8_t *Buffer = new uint8_t[maxSize];
  for (auto &KV : MemManager.AllocatedMemory) {
    std::cout << "[MemoryManager] Writing dev Memory: " << KV.first
              << " of Size: " << (uint64_t)KV.second.Size << "\n";
    PREFIX(Memcpy)
    ((void *)Buffer, (void *)KV.first, KV.second.Size,
     PREFIX(MemcpyDeviceToHost));
    os << llvm::StringRef(reinterpret_cast<const char *>(&KV.first),
                          sizeof(KV.first));
    if (os.has_error()) {
      std::ostringstream oss;
      oss << "Error when serializing device memory " << os.error();
      delete[] Buffer;
      throw std::runtime_error(oss.str());
    }

    os << llvm::StringRef(reinterpret_cast<const char *>(&KV.second.Size),
                          sizeof(KV.second.Size));
    if (os.has_error()) {
      std::ostringstream oss;
      oss << "Error when serializing device memory " << os.error();
      delete[] Buffer;
      throw std::runtime_error(oss.str());
    }

    os << llvm::StringRef(reinterpret_cast<const char *>(Buffer),
                          KV.second.Size);
    if (os.has_error()) {
      std::ostringstream oss;
      oss << "Error when serializing device memory " << os.error();
      delete[] Buffer;
      throw std::runtime_error(oss.str());
    }
  }

  delete[] Buffer;
  return os;
}
