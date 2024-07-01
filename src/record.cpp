#include "memory.hpp"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

using namespace llvm;

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

#define DEVICE_FUNC(x) "cuda" x
#define PREFIX_UU(x) __cuda##x
#define PREFIX(x) cuda##x

#elif defined(ENABLE_HIP)

#include <hip/hip_runtime.h>

#define DEVICE_FUNC(x) "hip" x
#define PREFIX_UU(x) __hip##x
#define PREFIX(x) hip##x

#else
#error "Must define ENABLE_CUDA or ENABLE_HIP when building"
#endif

#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#if ENABLE_REPLAY_OPT
#include "replay_opt.hpp"
#endif

// Overloaded functions
static PREFIX(Error_t) (*deviceLaunchKernelInternal)(
    const void *func, dim3 gridDim, dim3 blockDim, void **args,
    size_t sharedMem, PREFIX(Stream_t) stream) = nullptr;
static PREFIX(Error_t) (*deviceMallocInternal)(void **ptr, size_t size);
static PREFIX(Error_t) (*deviceMallocHostInternal)(void **ptr, size_t size);
static PREFIX(Error_t) (*deviceMallocManagedInternal)(void **ptr, size_t size,
                                                      unsigned int flags);
static PREFIX(Error_t) (*deviceFreeInternal)(void *devPtr);
static void (*__deviceRegisterVarInternal)(void **fatDevbinHandle,
                                           char *hostVar, char *deviceAddress,
                                           const char *deviceName, int ext,
                                           size_t size, int constant,
                                           int global);
static void **(*__deviceRegisterFatBinaryInternal)(void *fatDevbin);
static void (*__deviceRegisterFunctionInternal)(
    void **fatDevbinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);

static void (*__deviceRegisterFatBinaryEndInternal)(void *);

struct MappedAlloc {
  size_t actual_size;
  size_t requested_size;
  CUmemGenericAllocationHandle handle;
};

// Reverse eng. fat binary format, unused for now.
struct CudaRegisterFatBinaryArguments {
  int magic_number;
  int version;
  const void *binary;
  void *unused;
};

struct MeasureInfo {
  dim3 GridDim;
  dim3 BlockDim;
  double Time;
};

struct FuncInfo {
  uint64_t elements;
  void *ptr;
};

struct HostFuncInfo {
  uint64_t elements;
  void *dev_ptr;
  uint64_t *h_ptr;
  HostFuncInfo() : elements(0), dev_ptr(nullptr), h_ptr(nullptr) {}
  HostFuncInfo(FuncInfo &info) : elements(info.elements), dev_ptr(info.ptr) {
    h_ptr = nullptr;
    if (elements != 0) {
      h_ptr = new uint64_t[elements];
    }
  }

  void dump(bool detail = false) const {
    std::cout << "Host Pointer: " << h_ptr;
    std::cout << " Device Pointer: " << h_ptr;
    std::cout << " Number of elements: " << elements << "\n";
    if (detail) {
      for (int i = 0; i < elements; i++) {
        std::cout << "Element [" << i << "]" << h_ptr[i] << "\n";
      }
    }
  }

  ~HostFuncInfo() {}
};

// Singleton class that initializes wrapping.
class Wrapper {
public:
  static Wrapper *instance() {
    static Wrapper w;
    return &w;
  };

  void getDeviceInfo(int &WarpSize, int &MultiProcessorCount,
                     int &MaxGridSizeX) {
    WarpSize = this->WarpSize;
    MultiProcessorCount = this->MultiProcessorCount;
    MaxGridSizeX = this->MaxGridSizeX;

    return;
  }

  bool isTunable(const void *Func, int &MaxThreadsPerBlock, int Dim) {
    PREFIX(FuncAttributes) Attr;
    PREFIX(FuncGetAttributes)(&Attr, Func);
    DEBUG(printf("deviceFuncGetAttributes sharedSizeBytes %zu numRegs %d "
                 "maxThreadsPerBlock %d\n",
                 Attr.sharedSizeBytes, Attr.numRegs, Attr.maxThreadsPerBlock);)
    MaxThreadsPerBlock = Attr.maxThreadsPerBlock;

    // TODO: Assumes that using shared memory enforces a hard limit on
    // dimensions.
    bool IsSharedZero = (Attr.sharedSizeBytes == 0);

    return IsSharedZero;
  }

  template <typename dType>
  HostFuncInfo loadSymbolToHost(CUmodule &CUMod,
                                const std::string &symbolName) {
    CUdeviceptr DevPtr;
    size_t Bytes;

    if (cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, symbolName.c_str()) !=
        CUDA_SUCCESS)
      throw std::runtime_error("Cannot load Global " + symbolName + "\n");

    assert(Bytes == sizeof(FuncInfo) && "Func info size do not match");
    FuncInfo funcData;

    cudaErrCheck(PREFIX(Memcpy)(&funcData, (void *)DevPtr, Bytes,
                                PREFIX(MemcpyDeviceToHost)));

    std::cout << "Data Size : " << funcData.elements << " " << funcData.ptr
              << "\n";

    HostFuncInfo hfuncData(funcData);
    cudaErrCheck(PREFIX(Memcpy)(hfuncData.h_ptr, hfuncData.dev_ptr,
                                sizeof(dType) * hfuncData.elements,
                                PREFIX(MemcpyDeviceToHost)));
    return hfuncData;
  }

  void loadRRGlobals() {
    for (auto FB : FatBinaries) {
      CUmodule CUMod;
      std::cout << "Trying to open file " << FB.first << "\n";
      auto err = cuModuleLoadFatBinary(&CUMod, FB.first->binary);
      if (err != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot Open Module Binary" +
                                 std::to_string(static_cast<int>(err)));
      }
      for (auto GM : GlobalsMap) {
        if (GM.first.find("_record_replay_func_info_") != std::string::npos) {
          auto hfuncData = loadSymbolToHost<uint64_t>(CUMod, GM.first);
          std::cout << "Adding: " << GM.first << "\n";
          ArgsInfo[GM.first] = hfuncData;
          ArgsInfo[GM.first].dump(true);
        } else if (GM.first.find("_record_replay_descr_") !=
                   std::string::npos) {
          auto llvmIR = loadSymbolToHost<uint8_t>(CUMod, GM.first);
          std::error_code EC;
          std::string rrBC("record-ir.bc");
          raw_fd_ostream OutBC(rrBC, EC);
          if (EC)
            throw std::runtime_error("Cannot open device code " + rrBC);
          OutBC << StringRef(reinterpret_cast<const char *>(llvmIR.h_ptr),
                             llvmIR.elements);
          OutBC.close();
          llvmIR.dump();
        }
      }
    }
  }

  // GlobalsMap stores a mapping of host addr to size, will convert to device
  // pointers using <cuda|hip>GetSymbolAddress.
  std::unordered_map<std::string, size_t> GlobalsMap;
  // DeviceMemoryMap holds all the pointers to device memory that need to be
  // copied/restored.
  std::unordered_map<const void *, std::pair<std::string, std::string>>
      SymbolTable;
  std::vector<std::string> SymbolWhiteList;
  std::vector<std::string> SymbolExclList;
  std::vector<std::pair<CudaRegisterFatBinaryArguments *, uint64_t>>
      FatBinaries;

  std::unordered_map<std::string, HostFuncInfo> ArgsInfo;
  std::unordered_map<const void *, MappedAlloc> devMemory;

private:
  void *device_runtime_handle;
  int WarpSize;
  int MultiProcessorCount;
  int MaxGridSizeX;

  Wrapper() {
#ifdef ENABLE_CUDA
    device_runtime_handle = dlopen("libcudart.so", RTLD_NOW);
#else
    device_runtime_handle = dlopen("libamdhip64.so", RTLD_NOW);
#endif
    assert(device_runtime_handle && "Expected non-null");

    // Redirect overloaded device runtime functions.
    reinterpret_cast<void *&>(deviceLaunchKernelInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("LaunchKernel"));
    assert(deviceLaunchKernelInternal && "Expected non-null");
#if !defined(ENABLE_REPLAY_OPT)
    reinterpret_cast<void *&>(deviceMallocInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("Malloc"));
    assert(deviceMallocInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceMallocHostInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("MallocHost"));
    assert(deviceMallocHostInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceMallocManagedInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("MallocManaged"));
    assert(deviceMallocManagedInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceFreeInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("Free"));
    assert(deviceFreeInternal && "Expected non-null");

    reinterpret_cast<void *&>(__deviceRegisterVarInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterVar"));
    assert(__deviceRegisterVarInternal && "Expected non-null");
    reinterpret_cast<void *&>(__deviceRegisterFatBinaryInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFatBinary"));
    assert(__deviceRegisterFatBinaryInternal && "Expected non-null");
    reinterpret_cast<void *&>(__deviceRegisterFatBinaryEndInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFatBinaryEnd"));
    assert(__deviceRegisterFatBinaryEndInternal && "Expected non-null");

    reinterpret_cast<void *&>(__deviceRegisterFunctionInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFunction"));
    assert(__deviceRegisterFunctionInternal && "Expected non-null");
    DEBUG(printf("=== Library inited\n");)

    // Gather device info.
    int DeviceId;
    PREFIX(GetDevice(&DeviceId));
    PREFIX(DeviceGetAttribute)
    (&WarpSize,
    // NOTE: very ugly, thank you vendors!
#ifdef ENABLE_CUDA
     cudaDevAttrWarpSize,
#else
     hipDeviceAttributeWarpSize,
#endif
     DeviceId);
    PREFIX(DeviceGetAttribute)
    (&MultiProcessorCount,
#ifdef ENABLE_CUDA
     cudaDevAttrMultiProcessorCount,
#else
     hipDeviceAttributeMultiprocessorCount,
#endif
     DeviceId);
    // TODO: We use the x-dimension for now, consider adding y,z if needed.
    PREFIX(DeviceGetAttribute)
    (&MaxGridSizeX,
#ifdef ENABLE_CUDA
     cudaDevAttrMaxGridDimX,
#else
     hipDeviceAttributeMaxGridDimX,
#endif
     DeviceId);

    // Gather the symbols whitelist.
    const char *EnvVarSymbols = std::getenv("LIBREPLAY_SYMBOLS");
    std::string Symbols = (EnvVarSymbols ? EnvVarSymbols : "");
    DEBUG(std::cout << "Symbols EnvVar " << Symbols << "\n");

    if (!Symbols.empty())
      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
        endpos = Symbols.find(',', pos);
        SymbolWhiteList.push_back(Symbols.substr(pos, endpos));
        DEBUG(std::cout << "Symbol: " << Symbols.substr(pos, endpos) << "\n");
        pos = endpos + 1;
      }

    // Gather the symbols exclusion list.
    EnvVarSymbols = std::getenv("LIBREPLAY_EXCL_SYMBOLS");
    Symbols = (EnvVarSymbols ? EnvVarSymbols : "");
    DEBUG(std::cout << "Symbols Excl. EnvVar " << Symbols << "\n");

    if (!Symbols.empty())
      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
        endpos = Symbols.find(',', pos);
        SymbolExclList.push_back(Symbols.substr(pos, endpos));
        DEBUG(std::cout << "Symbol excl.: " << Symbols.substr(pos, endpos)
                        << "\n");
        pos = endpos + 1;
      }
#endif
  }
};

// Overload implementations.
extern "C" {

void PREFIX_UU(RegisterFatBinaryEnd)(void *ptr) {
  Wrapper *W = Wrapper::instance();
  __deviceRegisterFatBinaryEndInternal(ptr);
  return;
}

void **PREFIX_UU(RegisterFatBinary)(void *fatCubin) {
  Wrapper *W = Wrapper::instance();
  void **ret = __deviceRegisterFatBinaryInternal(fatCubin);
  return ret;
}

void PREFIX_UU(RegisterVar)(void **fatCubinHandle, char *hostVar,
                            char *deviceAddress, const char *deviceName,
                            int ext, size_t size, int constant, int global) {
  Wrapper *W = Wrapper::instance();
  __deviceRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress,
                              deviceName, ext, size, constant, global);
  DEBUG(std::cout << "hostVar " << (void *)hostVar << " deviceAddr "
                  << (void *)deviceAddress << " deviceName " << deviceName
                  << " ext " << ext << " size " << size << " constant "
                  << constant << " global " << global << "\n";)

  if (constant)
    return;

  assert(W->GlobalsMap.count(hostVar) == 0 &&
         "Expected non-duplicate Entry in globals");
  W->GlobalsMap[std::string(deviceName)] = size;
}

void PREFIX_UU(RegisterFunction)(void **fatCubinHandle, const char *hostFun,
                                 char *deviceFun, const char *deviceName,
                                 int thread_limit, uint3 *tid, uint3 *bid,
                                 dim3 *bDim, dim3 *gDim, int *wSize) {
  Wrapper *W = Wrapper::instance();
  DEBUG(std::cout << "Register func hostFun " << (void *)hostFun
                  << " deviceFun " << deviceFun << " Name " << deviceName
                  << "\n");
  int status;
  W->SymbolTable[(void *)hostFun] =
      std::make_pair(std::string(deviceFun), std::string(deviceName));
  __deviceRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun,
                                   deviceName, thread_limit, tid, bid, bDim,
                                   gDim, wSize);
}

PREFIX(Error_t) PREFIX(Malloc)(void **ptr, size_t size) {
  Wrapper *W = Wrapper::instance();
  size_t actual_size;
  auto ret = memory::cuda::MemMapToDevice(ptr, nullptr, size, actual_size);
  W->devMemory[*ptr] = {actual_size, size, ret};
  return PREFIX(Success);
}

PREFIX(Error_t) PREFIX(MallocHost)(void **ptr, size_t size) {
  Wrapper *W = Wrapper::instance();
  auto ret = deviceMallocHostInternal(ptr, size);
  return ret;
}

PREFIX(Error_t)
PREFIX(MallocManaged)(void **ptr, size_t size, unsigned int flags) {
  Wrapper *W = Wrapper::instance();
  auto ret = deviceMallocManagedInternal(ptr, size, flags);
  return ret;
}

PREFIX(Error_t) PREFIX(Free)(void *devPtr) {
  Wrapper *W = Wrapper::instance();
  assert(W->devMemory.find(devPtr) != W->devMemory.end() &&
         "Call Free on device memory that is not being tracked");
  memory::cuda::MemoryUnMap(devPtr, W->devMemory[devPtr].handle,
                            W->devMemory[devPtr].actual_size);
  W->devMemory.erase(devPtr);
  return PREFIX(Success);
}

void dumpDeviceMemory(std::string fileName, HostFuncInfo &info, void **args) {
  std::error_code EC;
  raw_fd_ostream OutBC(fileName, EC);
  Wrapper *W = Wrapper::instance();
  if (EC)
    throw std::runtime_error("Cannot open device code " + fileName);
  size_t maxSize = 0;
  size_t overhead = 0;
  size_t bytes = 0;

  // First we serialize the information on the argument list
  OutBC << StringRef(reinterpret_cast<const char *>(&info.elements),
                     sizeof(info.elements));

  for (int i = 0; i < info.elements; i++) {
    OutBC << StringRef(reinterpret_cast<const char *>(&info.h_ptr[i]),
                       sizeof(uint64_t));
  }

  for (int i = 0; i < info.elements; i++) {
    // raw_fd_ostream does not behave well when streaming other types
    // without specifying the size. I cast everything into a StringRef to be
    // explicit about the size of the bytes to be stored.
    OutBC << StringRef(reinterpret_cast<const char *>(args[i]), info.h_ptr[i]);
  }

  for (auto KV : W->devMemory) {
    if (maxSize < KV.second.requested_size)
      maxSize = KV.second.requested_size;
  }

  if (maxSize == 0)
    return;

  uint8_t *Buffer = new uint8_t[maxSize];
  for (auto KV : W->devMemory) {
    std::cout << "Writing dev Memory: " << KV.first
              << " of Size: " << KV.second.requested_size << "\n";
    PREFIX(Memcpy)
    ((void *)Buffer, (void *)KV.first, KV.second.requested_size,
     PREFIX(MemcpyDeviceToHost));
    OutBC << StringRef(reinterpret_cast<const char *>(&KV.first),
                       sizeof(KV.first));
    if (EC)
      throw std::runtime_error("Cannot open device code " + fileName);

    OutBC << StringRef(
        reinterpret_cast<const char *>(&KV.second.requested_size),
        sizeof(KV.second.requested_size));
    if (EC)
      throw std::runtime_error("Cannot open device code " + fileName);

    OutBC << StringRef(reinterpret_cast<const char *>(Buffer),
                       KV.second.requested_size);
    if (EC)
      throw std::runtime_error("Cannot open device code " + fileName);
  }

  OutBC.flush();
  OutBC.close();

  delete[] Buffer;
}

PREFIX(Error_t)
PREFIX(LaunchKernel)
(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
 PREFIX(Stream_t) stream) {
  Wrapper *W = Wrapper::instance();
  W->loadRRGlobals();
  std::string func_name = W->SymbolTable[func].first;
  func_name = "_record_replay_func_info_" + func_name;
  std::cout << "Looking for func: " << func_name << "\n";
  auto func_info = W->ArgsInfo[func_name];
  func_info.dump(true);
  dumpDeviceMemory("KernelBefore.bin", func_info, args);
  static std::size_t kernel_counter{0};
  PREFIX(Error_t) ret;

  printf("deviceLaunchKernel func %p name %s args %p sharedMem %zu\n", func,
         W->SymbolTable[func].first.c_str(), args, sharedMem);
  int MaxThreadsPerBlock;
  auto Dim =
      blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z * gridDim.z;

  cudaErrCheck(deviceLaunchKernelInternal(func, gridDim, blockDim, args,
                                          sharedMem, stream));
  // TODO: We need here to sync, make sure we get all the data after termination
  // of the kernel. Alternatively we can synchronize with the stream. I am
  // hesitant for this. Multi-stream execution can potentially modify the device
  // memory as we copy the data.
  PREFIX(DeviceSynchronize());
  dumpDeviceMemory("KernelAfter.bin", func_info, args);
  return ret;
}

void __rr_register_fat_binary(void *FatBinary, uint64_t size) {
  printf("Binary at address %p with size %ld\n", FatBinary, size);
  Wrapper *W = Wrapper::instance();
  W->FatBinaries.push_back(std::make_pair(
      static_cast<CudaRegisterFatBinaryArguments *>(FatBinary), size));
}
}
