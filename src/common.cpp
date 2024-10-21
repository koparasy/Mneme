#include "common.hpp"
#include "memory.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>

using namespace util;
GlobalVar::GlobalVar(GlobalVar &&old) {
  this->Name = old.Name;
  this->Size = old.Size;
  this->DevPtr = old.DevPtr;
  this->HostPtr = old.HostPtr;
  old.HostPtr = nullptr;
}

GlobalVar::GlobalVar(std::string Name, size_t Size, void *DevPtr)
    : Name(Name), Size(Size), DevPtr(DevPtr) {
  HostPtr = new uint8_t[Size];
}

GlobalVar::~GlobalVar() {
  if (HostPtr != nullptr)
    delete[] (uint8_t *)HostPtr;
}

GlobalVar &GlobalVar::operator=(GlobalVar &&other) noexcept {
  if (this != &other) {
    this->Name = other.Name;
    this->Size = other.Size;
    this->DevPtr = other.DevPtr;
    this->HostPtr = other.HostPtr;
    other.HostPtr = nullptr;
  }
  return *this;
}

GlobalVar::GlobalVar(const void **BufferPtr) {
  size_t name_size = *(size_t *)(*BufferPtr);
  *BufferPtr = advanceVoidPtr(*BufferPtr, sizeof(name_size));
  Name = std::string(static_cast<const char *>(*BufferPtr), name_size);
  *BufferPtr = advanceVoidPtr(*BufferPtr, name_size);
  Size = *(size_t *)*BufferPtr;
  *BufferPtr = advanceVoidPtr(*BufferPtr, sizeof(this->Size));
  HostPtr = new uint8_t[this->Size];
  std::memcpy(HostPtr, *BufferPtr, Size);
  *BufferPtr = advanceVoidPtr(*BufferPtr, Size);
}

void GlobalVar::dump() {
  //  std::cout << "Global Variable:" << Name << " " << Size << "\n";
  //  for (int i = 0; i < Size; i++) {
  //    std::cout << "Value at " << i << " is : " << (int)((int8_t *)HostPtr)[i]
  //              << "\n";
  //  }
}

void GlobalVar::setDevPtrFromModule(gpu::DeviceModule &Mod) {
  size_t LSize;
  gpu::DevicePtr LDevPtr;
  if (DRIVER_PREFIX(ModuleGetGlobal)(&LDevPtr, &LSize, Mod, Name.c_str()) !=
      gpu::DeviceSuccess)
    throw std::runtime_error("Cannot load Global " + Name + "\n");
  if (LSize != Size)
    throw std::runtime_error(
        "Global: " + Name +
        "differs in size between recording and replaying (Recorded:" +
        std::to_string(LSize) + " , Replayed: " + std::to_string(Size) + "\n");
  DevPtr = (void *)LDevPtr;
  Size = LSize;
}

void GlobalVar::copyToDevice() {
  DeviceRTErrCheck(
      PREFIX(Memcpy)(DevPtr, HostPtr, Size, PREFIX(MemcpyHostToDevice)));
}

void GlobalVar::copyFromDevice() {
  DeviceRTErrCheck(
      PREFIX(Memcpy)(HostPtr, DevPtr, Size, PREFIX(MemcpyDeviceToHost)));
}

bool GlobalVar::compare(GlobalVar &other) {
  if (Size != other.Size)
    return false;

  if (Name != other.Name)
    return false;

  if (std::memcmp(HostPtr, other.HostPtr, Size) != 0) {
    int8_t *host_ptr = (int8_t *)HostPtr;
    int8_t *ohost_ptr = (int8_t *)other.HostPtr;
    for (int i = 0; i < Size; i++) {
      DEBUG(std::cout << "Global Mem at address " << DevPtr << " at index:" << i
                      << " has the value of " << (int)host_ptr[i] << " versus "
                      << (int)ohost_ptr[i] << "\n";)
    }
    return false;
  }
  return true;
}
