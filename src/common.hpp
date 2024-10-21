#pragma once
#include <string>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include "macro.hpp"

namespace util {
template <typename Ty> inline Ty *advanceVoidPtr(Ty *Ptr, int64_t Offset) {
  static_assert(std::is_void<Ty>::value);
  return const_cast<char *>(reinterpret_cast<const char *>(Ptr) + Offset);
}

template <typename Ty> Ty RoundUp(Ty Size, Ty Divider) {
  return (Size + Divider - 1) & ~(Divider - 1);
}
} // namespace util

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

#ifdef ENABLE_CUDA
  void setDevPtrFromModule(CUmodule &Module);
#elif defined(ENABLE_HIP)
  void setDevPtrFromModule(hipModule_t &Module);
#endif

  void dump();

  void copyToDevice();
  void copyFromDevice();

  bool compare(GlobalVar &other);
};
