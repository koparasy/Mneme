#include "memory.hpp"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <map>
#include <regex>
#include <sys/resource.h>
#include <sys/time.h>

#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.hpp"
#include "macro.hpp"
#include <filesystem>

using namespace llvm;

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
#error "Must define ENABLE_CUDA or ENABLE_HIP when building"
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
#ifdef __ENABLE_DEBUG__
    std::cout << "Host Pointer: " << h_ptr;
    std::cout << " Device Pointer: " << h_ptr;
    std::cout << " Number of elements: " << elements << "\n";
    if (detail) {
      for (int i = 0; i < elements; i++) {
        std::cout << "Element [" << i << "]" << h_ptr[i] << "\n";
      }
    }
#endif
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
  HostFuncInfo loadSymbolToHost(void *DevPtr, size_t Bytes) {

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

  void addGlobal(const std::string &SymbolName, void *DevPtr, size_t Bytes) {
    if (SymbolName.find("_record_replay_func_info_") != std::string::npos) {
      auto hfuncData = loadSymbolToHost<uint64_t>(DevPtr, Bytes);
      auto FuncName = std::regex_replace(
          SymbolName, std::regex("_record_replay_func_info_"), "");

      std::cout << "Adding: " << FuncName << "\n";
      ArgsInfo[FuncName] = hfuncData;
      ArgsInfo[FuncName].dump(true);
    } else if (SymbolName.find("_record_replay_descr_") != std::string::npos) {
      std::string ModuleName = std::regex_replace(
          SymbolName, std::regex("_record_replay_descr_"), "");
      std::cout << "Found record replay description\n";
      auto llvmIRInfo = loadSymbolToHost<uint8_t>(DevPtr, Bytes);
      std::error_code EC;

      std::string extracted_ir_fn(
          Twine(record_replay_dir.string() + ModuleName, ".bc").str());
      raw_fd_ostream OutBC(extracted_ir_fn, EC);
      if (EC)
        throw std::runtime_error("Cannot open device code " + extracted_ir_fn);
      OutBC << StringRef(reinterpret_cast<const char *>(llvmIRInfo.h_ptr),
                         llvmIRInfo.elements);
      std::cout << "Registered Record replay descr";
      OutBC.close();
      llvmIRInfo.dump();
      std::cout << "One more module file " << extracted_ir_fn << "\n";
      ModuleFiles.push_back(std::move(extracted_ir_fn));
    } else {
      std::cout << "Device Address of Symbol " << SymbolName << " is " << DevPtr
                << " with size: " << Bytes << "\n";

      TrackedGlobalVars.emplace(
          SymbolName, std::move(GlobalVar(SymbolName, Bytes, (void *)DevPtr)));
    }

    // We erase here. This is not a global we would like to track
  }

  void loadRRGlobals() {
    static bool RRGlobalsInitialized = false;
    if (RRGlobalsInitialized)
      return;
    for (auto GM = GlobalsMap.begin(); GM != GlobalsMap.end(); GM++) {
      void *DevPtr;
      cudaErrCheck(cudaGetSymbolAddress(&DevPtr, GM->second.second));
      addGlobal(GM->first, DevPtr, GM->second.first);
    }
    RRGlobalsInitialized = true;
  }

  // Contains name of global and respective size;
  std::unordered_map<std::string, std::pair<size_t, const char *>> GlobalsMap;

  // All globals we will need to record.
  std::unordered_map<std::string, GlobalVar> TrackedGlobalVars;
  std::unordered_map<const void *, std::pair<std::string, std::string>>
      SymbolTable;
  std::vector<std::string> SymbolWhiteList;
  std::vector<std::string> SymbolExclList;
  std::vector<std::pair<CudaRegisterFatBinaryArguments *, uint64_t>>
      FatBinaries;

  std::unordered_map<std::string, HostFuncInfo> ArgsInfo;
  json::Object RecordedKernels;
  json::Array ModuleFiles;

  const std::filesystem::path &getDataStoreDir() const {
    return record_replay_dir;
  }

public:
  MemoryManager *MemManager;

private:
  void *device_runtime_handle;
  int WarpSize;
  int MultiProcessorCount;
  int MaxGridSizeX;
  std::string record_replay_fn;
  std::filesystem::path record_replay_dir;

  Wrapper() {
    MemManager = nullptr;
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
    auto env_rr_file = std::getenv("RR_FILE");
    if (!env_rr_file) {
      env_rr_file = (char *)"record_replay.json";
    }

    record_replay_fn = std::string(env_rr_file);

    auto env_rr_data_directory = std::getenv("RR_DATA_DIR");
    if (!env_rr_data_directory) {
      env_rr_data_directory = (char *)"./";
    }

    record_replay_dir = std::string(env_rr_data_directory);
    if (!std::filesystem::is_directory(record_replay_dir)) {
      throw std::runtime_error("Path :" + record_replay_dir.string() +
                               " does not exist.\n");
    }
  }

  ~Wrapper() {
    if (!MemManager)
      return;

    uintptr_t startAddr = MemManager->StartVAAddr();
    uint64_t totalSize = MemManager->TotalVASize();

    auto JsonFilename = record_replay_dir / record_replay_fn;
    std::error_code EC;
    json::Object record;
    std::ostringstream oss;
    oss << std::hex << startAddr;

    record["StartVAAddr"] = json::Value(oss.str());
    record["TotalSize"] = json::Value(totalSize);
    record["Kernels"] = json::Value(std::move(RecordedKernels));
    record["Modules"] = json::Value(
        std::move(ModuleFiles)); // json::Value(std::move(FuncsInModules));
    raw_fd_ostream JsonOS(JsonFilename.string(), EC);
    JsonOS << json::Value(std::move(record));
    JsonOS.close();

    if (MemManager) {
      delete MemManager;
      MemManager = nullptr;
    }
  }
};

// Overload implementations.
extern "C" {

// void PREFIX_UU(RegisterFatBinaryEnd)(void *ptr) {
//   Wrapper *W = Wrapper::instance();
//   __deviceRegisterFatBinaryEndInternal(ptr);
//   return;
// }
//
// void **PREFIX_UU(RegisterFatBinary)(void *fatCubin) {
//   Wrapper *W = Wrapper::instance();
//   return __deviceRegisterFatBinaryInternal(fatCubin);
// }

void PREFIX_UU(RegisterVar)(void **fatCubinHandle, char *hostVar,
                            char *deviceAddress, const char *deviceName,
                            int ext, size_t size, int constant, int global) {
  Wrapper *W = Wrapper::instance();
  __deviceRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress,
                              deviceName, ext, size, constant, global);

  if (constant)
    return;

  //  assert(W->GlobalsMap.count(hostVar) == 0 &&
  //         "Expected non-duplicate Entry in globals");
  W->GlobalsMap[std::string(deviceName)] = std::make_pair(size, hostVar);

  DEBUG(std::cout << "hostVar " << (void *)hostVar << " deviceAddr "
                  << (void *)deviceAddress << " deviceName " << deviceName
                  << " ext " << ext << " size " << size << " constant "
                  << constant << " global " << global << " SymbolAddr is "
                  << "\n";)
  //  W->addGlobal(deviceName, DevPtr, size);
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

void *suggestAddr() {
  // FIXME: This was a try and error approach. Getting a device address this
  // way allows replay to obtain it again. Otherwise driver returns
  // insignficant results
  void *ptr;
  cudaErrCheck(deviceMallocInternal(&ptr, 1024));
  std::cout << "Suggested Address " << std::hex << ptr << "\n";
  cudaErrCheck(deviceFreeInternal(ptr));
  return ptr;
}

PREFIX(Error_t) PREFIX(Malloc)(void **ptr, size_t size) {
  static long count = 0;

  Wrapper *W = Wrapper::instance();
  if (W->MemManager == nullptr) {
    void *initialAddr = suggestAddr();
    std::cout << "Initializing memory manager\n";
    W->MemManager =
        new MemoryManager(12L * 1024L * 1024L * 1024L, initialAddr, 0);
    std::cout << "Done \n";
  }
  *ptr = W->MemManager->allocate(size);
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
  assert(W->MemManager != nullptr &&
         "When Freeing memory Memory Manager needs to be initialized\n");
  W->MemManager->release(devPtr);
  return PREFIX(Success);
}

void dumpDeviceMemory(std::string fileName, HostFuncInfo &info, void **args,
                      std::unordered_map<std::string, GlobalVar> &globalVars) {
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
    std::cout << "Argument " << i << " Address "
              << (void *)(*(uint64_t *)(args[i])) << "Size:" << info.h_ptr[i]
              << "\n";
    OutBC << StringRef(reinterpret_cast<const char *>(args[i]), info.h_ptr[i]);
  }

  size_t num_variables = globalVars.size();
  std::cout << "Adding " << num_variables << " global variables\n";
  OutBC << StringRef(reinterpret_cast<const char *>(&num_variables),
                     sizeof(num_variables));

  for (const auto &KV : globalVars) {
    const auto &GV = KV.second;

    size_t name_size = GV.Name.size();
    OutBC << StringRef(reinterpret_cast<const char *>(&name_size),
                       sizeof(name_size));
    OutBC << StringRef(GV.Name.c_str(), GV.Name.size());
    OutBC << StringRef(reinterpret_cast<const char *>(&GV.Size),
                       sizeof(GV.Size));

    PREFIX(Memcpy)
    ((void *)GV.HostPtr, (void *)GV.DevPtr, GV.Size,
     PREFIX(MemcpyDeviceToHost));
    OutBC << StringRef(reinterpret_cast<const char *>(GV.HostPtr), GV.Size);
  }
  OutBC << *W->MemManager;

  OutBC.flush();
  OutBC.close();
}

PREFIX(Error_t)
PREFIX(LaunchKernel)
(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
 PREFIX(Stream_t) stream) {
  Wrapper *W = Wrapper::instance();
  W->loadRRGlobals();
  std::string func_name = W->SymbolTable[func].first;
  auto func_info = W->ArgsInfo[func_name];
  func_info.dump(true);
  PREFIX(Error_t) ret;

  printf("deviceLaunchKernel func %p name %s args %p sharedMem %zu\n", func,
         W->SymbolTable[func].first.c_str(), args, sharedMem);

  json::Object KernelInfo;
  KernelInfo["Name"] = W->SymbolTable[func].first.c_str();
  json::Object Grid, Block;
  Grid["x"] = gridDim.x;
  Grid["y"] = gridDim.y;
  Grid["z"] = gridDim.z;

  Block["x"] = blockDim.x;
  Block["y"] = blockDim.y;
  Block["z"] = blockDim.z;

  auto iDataFn = W->getDataStoreDir() /
                 Twine("Before" + W->SymbolTable[func].first + ".bin").str();

  dumpDeviceMemory(iDataFn.string(), func_info, args, W->TrackedGlobalVars);
  func_info.dump(true);
  KernelInfo["InputData"] = iDataFn.string();

  KernelInfo["Grid"] = json::Value(std::move(Grid));
  KernelInfo["Block"] = json::Value(std::move(Block));
  KernelInfo["SharedMemory"] = sharedMem;

  cudaErrCheck(deviceLaunchKernelInternal(func, gridDim, blockDim, args,
                                          sharedMem, stream));
  // TODO: We need here to sync, make sure we get all the data after
  // termination of the kernel. Alternatively we can synchronize with the
  // stream. I am hesitant for this. Multi-stream execution can potentially
  // modify the device memory as we copy the data.
  PREFIX(DeviceSynchronize());
  auto oDataFn = W->getDataStoreDir() /
                 Twine("After" + W->SymbolTable[func].first + ".bin").str();
  KernelInfo["OutputData"] = oDataFn.string();

  func_info.dump(true);
  dumpDeviceMemory(oDataFn.string(), func_info, args, W->TrackedGlobalVars);
  func_info.dump(true);
  W->RecordedKernels[func_name] = std::move(KernelInfo);

  return ret;
}

void __rr_register_fat_binary(void *FatBinary, uint64_t size) {
  printf("Binary at address %p with size %ld\n", FatBinary, size);
  Wrapper *W = Wrapper::instance();
  W->FatBinaries.push_back(std::make_pair(
      static_cast<CudaRegisterFatBinaryArguments *>(FatBinary), size));
}
}
