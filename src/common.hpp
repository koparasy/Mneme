#pragma once
#include <string>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "macro.hpp"

template <typename Ty> inline Ty *advanceVoidPtr(Ty *Ptr, int64_t Offset) {
  static_assert(std::is_void<Ty>::value);
  return const_cast<char *>(reinterpret_cast<const char *>(Ptr) + Offset);
}

struct GlobalVar {
  std::string Name;
  size_t Size;
  void *DevPtr;
  void *HostPtr;

  GlobalVar() = delete;
  GlobalVar(const GlobalVar &) = delete;
  GlobalVar &operator=(const GlobalVar &) = delete;

  GlobalVar(GlobalVar &&old);
  GlobalVar &operator=(GlobalVar &&other) noexcept;

  explicit GlobalVar(std::string Name, size_t Size, void *DevPtr);
  explicit GlobalVar(const void **BufferPtr);
  ~GlobalVar();

  void setDevPtrFromModule(CUmodule &Module);

  void dump();

  void copyToDevice();
  void copyFromDevice();

  bool compare(GlobalVar &other);
};
