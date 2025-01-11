#pragma once
#include "macro.hpp"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <set>
#include <sys/types.h>

struct ContiguousAddrBlock {
  // Starting address of the free block
  uintptr_t PageAddr;
  // Size of the free block
  uint64_t Size;

  ContiguousAddrBlock(uintptr_t start, uint64_t sz);

  // Comparison operators for sorting blocks by address and size
  bool operator<(const ContiguousAddrBlock &other) const;
};

class MemoryManager;
class PageManager {
  friend MemoryManager;

protected:
  std::multiset<ContiguousAddrBlock> FreeVARanges;
  uintptr_t ReservedVA;
  uint64_t TotalVASize;
  uint64_t PageSize;
  int32_t DeviceID;

  // Erase a block from the multiset
  bool EraseVirtualAddress(uintptr_t addr, size_t size);
  // Function to coalesce contiguous blocks
  void coalesce();

  // Find a block that has a size >= requestedSize
  std::multiset<ContiguousAddrBlock>::iterator
  findFreeBlock(size_t requestedSize);
  // Find a block that includes the range [Addr, Addr + size)

  std::multiset<ContiguousAddrBlock>::iterator findInclusivePage(uintptr_t Addr,
                                                                 size_t size);

  std::pair<uintptr_t, uint64_t> ReserveBestFitPage(uint64_t VASize);

  std::pair<uintptr_t, uint64_t> RequestExactPage(uint64_t VASize, void *VA);

  std::pair<uintptr_t, uint64_t> AllocatePage(uint64_t VASize, void *VA);

public:
  PageManager(uint64_t VASize, void *VA = nullptr, int32_t device_id = 0);
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
  gpu::MemoryBlob AllocateMemoryBlob(uint64_t VASize, void *VA = nullptr);

  void ReleaseMemoryBlob(gpu::MemoryBlob &&Blob);

  ~PageManager();
};

class MemoryManager {
  PageManager FreePages;

protected:
  std::unordered_map<void *, gpu::MemoryBlob> AllocatedMemory;

public:
  MemoryManager(uint64_t VASize, void *VA = nullptr, int32_t device_id = 0);

  void *allocate(uint64_t size, void *req_addr = nullptr);

  void release(void *addr);
  friend llvm::raw_fd_ostream &operator<<(llvm::raw_fd_ostream &os,
                                          const MemoryManager &MemManager);

  uintptr_t StartVAAddr() const;
  uint64_t TotalVASize() const;
};

llvm::raw_fd_ostream &operator<<(llvm::raw_fd_ostream &os,
                                 const MemoryManager &MemManager);
