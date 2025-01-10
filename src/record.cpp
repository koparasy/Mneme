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
#include <sys/types.h>
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
    DEBUG(std::cout << "Host Pointer: " << h_ptr;)
    DEBUG(std::cout << " Device Pointer: " << h_ptr;)
    DEBUG(std::cout << " Number of elements: " << elements << "\n";)
    if (detail) {
      for (int i = 0; i < elements; i++) {
        DEBUG(std::cout << "Element [" << i << "]" << h_ptr[i] << "\n";)
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

  bool RecordKernel(std::string &KernelName, dim3 &gridDim, dim3 blockDim) {

    long largestThreadID = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y) *
                           (gridDim.z * blockDim.z);
    auto KD = RecordedDims.find(KernelName);
    // If we have already recorded this kernel we know this is within our
    // whitelist
    if (KD != RecordedDims.end()) {
      if (KD->second < largestThreadID) {
        return true;
      }
      return false;
    }

    std::string all("all");
    auto it = std::find(SymbolWhiteList.begin(), SymbolWhiteList.end(), all);
    if (it != SymbolWhiteList.end())
      return true;

    for (auto Name : SymbolWhiteList) {
      if (KernelName.find(Name) != std::string::npos)
        return true;
    }

    return false;
  }

  void getDeviceInfo(int &WarpSize, int &MultiProcessorCount,
                     int &MaxGridSizeX) {
    WarpSize = this->WarpSize;
    MultiProcessorCount = this->MultiProcessorCount;
    MaxGridSizeX = this->MaxGridSizeX;

    return;
  }

  bool isTunable(const void *Func, int &MaxThreadsPerBlock, int Dim) {
    PREFIX(FuncAttributes) Attr;
    DeviceRTErrCheck(PREFIX(FuncGetAttributes)(&Attr, Func));
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

    DeviceRTErrCheck(PREFIX(Memcpy)(&funcData, (void *)DevPtr, Bytes,
                                    PREFIX(MemcpyDeviceToHost)));

    DEBUG(std::cout << "Data Size : " << funcData.elements << " "
                    << funcData.ptr << "\n";)

    HostFuncInfo hfuncData(funcData);
    DeviceRTErrCheck(PREFIX(Memcpy)(hfuncData.h_ptr, hfuncData.dev_ptr,
                                    sizeof(dType) * hfuncData.elements,
                                    PREFIX(MemcpyDeviceToHost)));
    return hfuncData;
  }

  void addGlobal(const std::string &SymbolName, void *DevPtr, size_t Bytes) {
    if (SymbolName.find("_record_replay_func_info_") != std::string::npos) {
      auto hfuncData = loadSymbolToHost<uint64_t>(DevPtr, Bytes);
      auto FuncName = std::regex_replace(
          SymbolName, std::regex("_record_replay_func_info_"), "");

      DEBUG(std::cout << "Adding: " << FuncName << "\n";)
      ArgsInfo[FuncName] = hfuncData;
      ArgsInfo[FuncName].dump(true);
    } else if (SymbolName.find("_record_replay_descr_") != std::string::npos) {
      std::string ModuleName = std::regex_replace(
          SymbolName, std::regex("_record_replay_descr_"), "");
      DEBUG(std::cout << "Found record replay description\n";)
      auto llvmIRInfo = loadSymbolToHost<uint8_t>(DevPtr, Bytes);
      std::error_code EC;

      std::string extracted_ir_fn(
          Twine(record_replay_dir.string() + ModuleName, ".bc").str());
      raw_fd_ostream OutBC(extracted_ir_fn, EC);
      if (EC)
        throw std::runtime_error("Cannot open device code " + extracted_ir_fn);
      OutBC << StringRef(reinterpret_cast<const char *>(llvmIRInfo.h_ptr),
                         llvmIRInfo.elements);
      DEBUG(std::cout << "Registered Record replay descr";)
      OutBC.close();
      llvmIRInfo.dump();
      DEBUG(std::cout << "One more module file " << extracted_ir_fn << "\n";)
      ModuleFiles.push_back(std::move(extracted_ir_fn));
    } else {
      DEBUG(std::cout << "Device Address of Symbol " << SymbolName << " is "
                      << DevPtr << " with size: " << Bytes << "\n";)

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
      DEBUG(std::cout << "Getting symbol address of symbol " << GM->first
                      << " with host address: " << (void *)GM->second.second
                      << "\n";)
      DeviceRTErrCheck(
          PREFIX(GetSymbolAddress)(&DevPtr, (const void *)GM->second.second));
      addGlobal(GM->first, DevPtr, GM->second.first);
      // hipError_t err = hipGetSymbolAddress(&DevPtr, GM->second.second);
    }
    RRGlobalsInitialized = true;
  }

  const std::filesystem::path &getDataStoreDir() const {
    return record_replay_dir;
  }

  uint64_t SuggestedSize() const { return VAMemSuggestedSize; }

public:
  MemoryManager *MemManager;
  // Contains name of global and respective size;
  std::unordered_map<std::string, std::pair<size_t, const char *>> GlobalsMap;

  // All globals we will need to record.
  std::unordered_map<std::string, GlobalVar> TrackedGlobalVars;
  std::unordered_map<const void *, std::pair<std::string, std::string>>
      SymbolTable;
  std::vector<std::string> SymbolWhiteList;
  std::vector<std::pair<CudaRegisterFatBinaryArguments *, uint64_t>>
      FatBinaries;

  std::unordered_map<std::string, HostFuncInfo> ArgsInfo;
  json::Object RecordedKernels;
  std::unordered_map<std::string, uint64_t> RecordedDims;
  json::Array ModuleFiles;

private:
  void *device_runtime_handle;
  int WarpSize;
  int MultiProcessorCount;
  int MaxGridSizeX;
  uint64_t VAMemSuggestedSize;
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

    reinterpret_cast<void *&>(deviceMallocInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("Malloc"));
    assert(deviceMallocInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceMallocHostInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("MallocHost"));
    assert(deviceMallocHostInternal && "Expect/ed non-null");
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
    DeviceRTErrCheck(PREFIX(GetDevice(&DeviceId)));
    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&WarpSize,
    // NOTE: very ugly, thank you vendors!
#ifdef ENABLE_CUDA
                                                cudaDevAttrWarpSize,
#else
                                                hipDeviceAttributeWarpSize,
#endif
                                                DeviceId));
    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&MultiProcessorCount,
#ifdef ENABLE_CUDA
                                                cudaDevAttrMultiProcessorCount,
#else
                                                hipDeviceAttributeMultiprocessorCount,
#endif
                                                DeviceId));
    // TODO: We use the x-dimension for now, consider adding y,z if needed.
    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&MaxGridSizeX,
#ifdef ENABLE_CUDA
                                                cudaDevAttrMaxGridDimX,
#else
                                                hipDeviceAttributeMaxGridDimX,
#endif
                                                DeviceId));

    // Gather the symbols whitelist.
    const char *EnvVarSymbols = std::getenv("RR_SYMBOLS");
    std::string Symbols = (EnvVarSymbols ? EnvVarSymbols : "all");

    if (!Symbols.empty()) {
      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
        endpos = Symbols.find(',', pos);
        SymbolWhiteList.push_back(Symbols.substr(pos, endpos));
        pos = endpos + 1;
      }
    }

    auto env_rr_file = std::getenv("RR_FILE");
    if (!env_rr_file) {
      env_rr_file = (char *)"record_replay.json";
    }

    record_replay_fn = std::string(env_rr_file);

    auto env_rr_data_directory = std::getenv("RR_DATA_DIR");
    record_replay_dir =
        (env_rr_data_directory ? std::string(env_rr_data_directory)
                               : std::filesystem::current_path().string());

    if (!std::filesystem::is_directory(record_replay_dir)) {
      throw std::runtime_error("Path :" + record_replay_dir.string() +
                               " does not exist.\n");
    }
    record_replay_dir = std::filesystem::absolute(record_replay_dir);

    const char *EnvVAMemSize = std::getenv("RR_VA_SIZE");
    VAMemSuggestedSize = ((EnvVAMemSize ? std::atol(EnvVAMemSize) : 12L)) *
                         1024L * 1024L * 1024L;
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

    delete MemManager;
    MemManager = nullptr;
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
  return __deviceRegisterFatBinaryInternal(fatCubin);
}

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
                  << constant << " global " << global << "\n";)
  //  W->addGlobal(deviceName, DevPtr, size);
}

void PREFIX_UU(RegisterFunction)(void **fatCubinHandle, const char *hostFun,
                                 char *deviceFun, const char *deviceName,
                                 int thread_limit, uint3 *tid, uint3 *bid,
                                 dim3 *bDim, dim3 *gDim, int *wSize) {
  Wrapper *W = Wrapper::instance();
  DEBUG(std::cout << "Register func hostFun " << (void *)hostFun
                  << " deviceFun " << deviceFun << " Name " << deviceName
                  << "\n";)
  int status;
  W->SymbolTable[(void *)hostFun] =
      std::make_pair(std::string(deviceFun), std::string(deviceName));
  __deviceRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun,
                                   deviceName, thread_limit, tid, bid, bDim,
                                   gDim, wSize);
}

void *suggestAddr() {
#ifdef ENABLE_CUDA
  // FIXME: This was a try and error approach. Getting a device address this
  // way allows replay to obtain it again. Otherwise driver returns
  // insignficant results
  void *ptr;
  DeviceRTErrCheck(deviceMallocInternal(&ptr, 1024));
  DEBUG(std::cout << "Suggested Address " << std::hex << ptr << "\n";)
  DeviceRTErrCheck(deviceFreeInternal(ptr));
  return ptr;
#elif defined(ENABLE_HIP)
  return nullptr;
#endif
}

PREFIX(Error_t) PREFIX(Malloc)(void **ptr, size_t size) {
  static long count = 0;

  Wrapper *W = Wrapper::instance();
  if (W->MemManager == nullptr) {
    void *initialAddr = suggestAddr();
    DEBUG(std::cout << "Initializing memory manager\n";)
    W->MemManager = new MemoryManager(W->SuggestedSize(), initialAddr, 0);
    DEBUG(std::cout << "Done \n";)
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
  std::cout << "Releasing memory " << std::hex << devPtr << "\n";
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
  if (EC) {
    throw std::runtime_error("File Error : " + EC.message());
  }
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
    DEBUG(std::cout << "Argument " << i << " Address "
                    << (void *)(*(uint64_t *)(args[i]))
                    << " Size:" << info.h_ptr[i] << "\n";)
    OutBC << StringRef(reinterpret_cast<const char *>(args[i]), info.h_ptr[i]);
  }

  size_t num_variables = globalVars.size();
  DEBUG(std::cout << "Adding " << num_variables << " global variables\n";)
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

    DeviceRTErrCheck(PREFIX(Memcpy)((void *)GV.HostPtr, (void *)GV.DevPtr,
                                    GV.Size, PREFIX(MemcpyDeviceToHost)));
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
  std::hash<std::string> hasher;
  std::string func_name = W->SymbolTable[func].first;
  PREFIX(Error_t) ret;

  long largestThreadID = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y) *
                         (gridDim.z * blockDim.z);

  printf("Kernel dims  ... %ld\n", largestThreadID);
  if (!W->RecordKernel(func_name, gridDim, blockDim)) {
    printf("Skipping \n");
    DeviceRTErrCheck(deviceLaunchKernelInternal(func, gridDim, blockDim, args,
                                                sharedMem, stream));
    return ret;
  }
  W->RecordedDims[func_name] = largestThreadID;

  printf("deviceLaunchKernel func %p name %s args %p sharedMem %zu\n", func,
         W->SymbolTable[func].first.c_str(), args, sharedMem);

  W->loadRRGlobals();
  auto func_info = W->ArgsInfo[func_name];
  func_info.dump(true);

  json::Object KernelInfo;
  KernelInfo["Name"] = W->SymbolTable[func].first.c_str();
  json::Object Grid, Block;
  Grid["x"] = gridDim.x;
  Grid["y"] = gridDim.y;
  Grid["z"] = gridDim.z;

  Block["x"] = blockDim.x;
  Block["y"] = blockDim.y;
  Block["z"] = blockDim.z;

  auto iDataFn =
      W->getDataStoreDir() /
      Twine("Before" + std::to_string(hasher(W->SymbolTable[func].first)) +
            ".bin")
          .str();

  dumpDeviceMemory(iDataFn.string(), func_info, args, W->TrackedGlobalVars);
  func_info.dump(true);
  KernelInfo["InputData"] = iDataFn.string();

  KernelInfo["Grid"] = json::Value(std::move(Grid));
  KernelInfo["Block"] = json::Value(std::move(Block));
  KernelInfo["SharedMemory"] = sharedMem;

  DeviceRTErrCheck(deviceLaunchKernelInternal(func, gridDim, blockDim, args,
                                              sharedMem, stream));
  // TODO: We need here to sync, make sure we get all the data after
  // termination of the kernel. Alternatively we can synchronize with the
  // stream. I am hesitant for this. Multi-stream execution can potentially
  // modify the device memory as we copy the data.
  DeviceRTErrCheck(PREFIX(DeviceSynchronize()));
  auto oDataFn =
      W->getDataStoreDir() /
      Twine("After" + std::to_string(hasher(W->SymbolTable[func].first)) +
            ".bin")
          .str();
  KernelInfo["OutputData"] = oDataFn.string();

  func_info.dump(true);
  dumpDeviceMemory(oDataFn.string(), func_info, args, W->TrackedGlobalVars);
  func_info.dump(true);
  W->RecordedKernels[func_name] = std::move(KernelInfo);

  return ret;
}
}

// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libRRPass.dylib -legacy-rr-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libRRPass.dylib -passes="rr-pass" `\`
//        -disable-output <input-llvm-file>
