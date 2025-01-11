#pragma once

#include "Logger.hpp"
#include "Utils.hpp"
#include <assert.h>
#include <cstddef>
#include <dlfcn.h>

#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StableHashing.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mutex>

#include "DeviceTraits.hpp"

namespace mneme {

struct FatBinaryWrapper_t {
  int Magic;
  int Version;
  const char *Binary;
  void **PrelinkedFatBins;
};

struct KernelInfo {
  const char *Name;
  llvm::SmallVector<size_t> KernelArgs;
  llvm::SmallVector<std::string> ModuleFiles;
  KernelInfo(char *Name) : Name(Name) {};
  KernelInfo() : Name(nullptr) {};

public:
  const char *getName() { return Name; }
  void setArgs(llvm::ArrayRef<size_t> ArgSizes) {
    KernelArgs = llvm::SmallVector<size_t>(ArgSizes);
  }
};

struct GlobalVarInfo {
  const char *Name;
  const void *HostSymbolAddr;
  const void *DevAddr;
  size_t VarSize;
  GlobalVarInfo(const char *Name, const void *HostSymbolAddr, size_t VarSize)
      : Name(Name), HostSymbolAddr(HostSymbolAddr), VarSize(VarSize),
        DevAddr(nullptr) {};
};

template <typename ImplT, typename MemBlobT> class MnemeRecorder {
protected:
  void *rtLib;
  std::string RecordReplayDir;
  llvm::DenseMap<void **, FatBinaryWrapper_t *> HandleToBin;
  llvm::DenseMap<void **, llvm::SmallVector<std::shared_ptr<KernelInfo>>>
      HandleToKernels;
  llvm::DenseMap<const void *, std::shared_ptr<KernelInfo>> KernelInfoMap;
  llvm::DenseMap<void **, llvm::SmallVector<GlobalVarInfo>>
      HandleToGlobalSymbol;
  llvm::DenseMap<void *, MemBlobT> AllocatedBlobs;

public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

private:
  bool ExtractedIR;
  std::once_flag ExtractFlag;

  DeviceError_t (*origLaunchKernel)(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem,
                                    DeviceStream_t stream) = nullptr;

  DeviceError_t (*origMallocDevice)(void **ptr, size_t size);

  DeviceError_t (*origMallocPinned)(void **ptr, size_t size,
                                    unsigned int flags);

  DeviceError_t (*origMallocManaged)(void **ptr, size_t size,
                                     unsigned int flags);

  DeviceError_t (*origFreeDevice)(void *devPtr);

  DeviceError_t (*origFreeHost)(void *ptr);

  void (*origRegisterDeviceVar)(void **fatbinHandle, char *hostVar,
                                char *deviceAddress, const char *deviceName,
                                int ext, size_t size, int constant, int global);

  void (*origRegisterFunction)(void **fatbinHandle, const char *hostFun,
                               char *deviceFun, const char *deviceName,
                               int thread_limit, uint3 *tid, uint3 *bid,
                               dim3 *bDim, dim3 *gDim, int *wSize);

  void **(*origRegisterFatBinary)(void *fatDevbin);

  void (*origRegisterFatBinaryEnd)(void *);

private:
  void extractIR() {
    std::cout << "I am here\n";
    static_cast<ImplT &>(*this).extractIR();
  }

public:
  void registerFatBinEnd(void *ptr) {
    DBG(Logger::logs("mneme") << "Registering FatBinaryEnd at address "
                              << std::hex << ptr << std::dec << "\n");
    origRegisterFatBinaryEnd(ptr);
  }

  void **registerFatBin(FatBinaryWrapper_t *fatbin) {
    void **Handle = origRegisterFatBinary(fatbin);
    DBG(Logger::logs("mneme")
        << "Registering FatBinary at address " << std::hex << fatbin << std::dec
        << " Return Ptr is: " << std::hex << Handle << std::dec << "\n");
    HandleToBin.insert({Handle, fatbin});
    for (auto &[H, B] : HandleToBin) {
      DBG(Logger::logs("mneme")
              << "Handle : " << std::hex << H << std::dec << " mapped to "
              << std::hex << B << std::dec << "\n";)
    }
    Logger::logs("mneme") << "Add of this is " << std::hex << this << "\n";

    return Handle;
  }

  void registerVar(void **fatBinHandle, char *hostVar, char *deviceAddress,
                   const char *deviceName, int ext, size_t size, int constant,
                   int global) {
    DBG(Logger::logs("mneme")
        << "Registering variable from handle " << std::hex << fatBinHandle
        << std::dec << " " << hostVar << "In address" << deviceName << " "
        << ext << " " << size << " " << constant << " " << global << "\n");
    origRegisterDeviceVar(fatBinHandle, hostVar, deviceAddress, deviceName, ext,
                          size, constant, global);
    if (!constant)
      HandleToGlobalSymbol[fatBinHandle].emplace_back(
          GlobalVarInfo(deviceName, hostVar, size));
    return;
  }

  void registerFunc(void **fatBinHandle, const char *hostFun, char *deviceFun,
                    const char *deviceName, int thread_limit, uint3 *tid,
                    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    DBG(Logger::logs("mneme")
        << "Registering Function from handle " << std::hex << fatBinHandle
        << std::dec << " HostFun:" << hostFun << " deviceFun:" << deviceFun
        << " deviceName:" << deviceName << " thread_limit:" << thread_limit
        << "\n");
    Logger::logs("mneme") << "Add of this is " << std::hex << this << "\n";
    if (!HandleToBin.contains(fatBinHandle))
      FATAL_ERROR("Handle container does not contain fatbin handle");
    std::shared_ptr<KernelInfo> KI = std::make_shared<KernelInfo>(deviceFun);
    KernelInfoMap.insert({(const void *)hostFun, KI});
    HandleToKernels[fatBinHandle].emplace_back(KI);
    origRegisterFunction(fatBinHandle, hostFun, deviceFun, deviceName,
                         thread_limit, tid, bid, bDim, gDim, wSize);
  };

  DeviceError_t rtMalloc(void **ptr, size_t size) {
    auto ret = origMallocDevice(ptr, size);
    DBG(Logger::logs("mneme") << "Malloced Device Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  };

  DeviceError_t rtManagedMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocManaged(ptr, size, flags);
    DBG(Logger::logs("mneme") << "Malloced Managed Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  };

  DeviceError_t rtHostMalloc(void **ptr, size_t size, unsigned int flags) {
    auto ret = origMallocPinned(ptr, size, flags);
    DBG(Logger::logs("mneme") << "Malloced Pinned Pointer " << *ptr
                              << " with size: " << size << "\n");
    return ret;
  }

  DeviceError_t rtFree(void *ptr) {
    auto ret = origFreeDevice(ptr);
    DBG(Logger::logs("mneme")
        << "Free Address " << std::hex << ptr << std::dec << "\n");
    return ret;
  };

  DeviceError_t rtHostFree(void *ptr) {
    auto ret = origFreeHost(ptr);
    DBG(Logger::logs("mneme")
        << "Free pinned address: " << std::hex << ptr << std::dec << "\n");
    return ret;
  }

  DeviceError_t rtLaunchKernel(const void *func, dim3 &GridDim, dim3 &BlockDim,
                               void **Args, size_t SharedMem,
                               DeviceStream_t Stream) {
    std::cout << "Entering " << &ExtractedIR << "\n";
    if (!KernelInfoMap.contains(func))
      FATAL_ERROR("Non registered kernel");

    std::cout << "My ID is " << this << "\n";
    std::call_once(ExtractFlag, [this]() { extractIR(); });

    auto &KInfo = KernelInfoMap[func];
    DBG(Logger::logs("mneme")
            << "Launching Kernel " << std::hex << func << std::dec << " KName"
            << KInfo->Name << " GDimX: " << GridDim.x << " GDimY: " << GridDim.y
            << " GDimZ: " << GridDim.z << " BDimX: " << BlockDim.x
            << " BDimY: " << BlockDim.y << " BDimZ: " << BlockDim.z << "\n";);
    auto ret =
        origLaunchKernel(func, GridDim, BlockDim, Args, SharedMem, Stream);
    return ret;
  }

  std::string storeModule(llvm::Module &M) {
    static int TotalModules = 0;
    std::error_code EC;
    std::string Filename(llvm::Twine(RecordReplayDir + "RecordedIR_" +
                                     std::to_string(TotalModules) + ".bc")
                             .str());
    llvm::raw_fd_ostream OutBC(Filename, EC);
    if (EC)
      FATAL_ERROR("Cannot write module ir file");

    OutBC << M;
    DBG(std::cout << "Registered Record replay descr");
    OutBC.close();
    return Filename;
  }

  MnemeRecorder() : ExtractedIR(true) {
    rtLib = ImplT::getRTLib();
    auto Dir = std::getenv("RR_DATA_DIR");
    if (Dir)
      RecordReplayDir = Dir;
    // MemManager = nullptr;

    // Redirect overloaded device runtime functions.
    reinterpret_cast<void *&>(origLaunchKernel) =
        dlsym(rtLib, ImplT::getLaunchKernelFnName());
    assert(origLaunchKernel &&
           "Expected non-null kernel-launch function pointer");

    reinterpret_cast<void *&>(origMallocDevice) =
        dlsym(rtLib, ImplT::getDeviceMallocFnName());
    assert(origMallocDevice &&
           "Expected non-null device malloc function pointer");

    reinterpret_cast<void *&>(origMallocPinned) =
        dlsym(rtLib, ImplT::getPinnedMallocFnName());
    assert(origMallocPinned &&
           "Expected non-null pinned malloc function pointer");

    reinterpret_cast<void *&>(origMallocManaged) =
        dlsym(rtLib, ImplT::getManagedMallocFnName());
    assert(origMallocManaged &&
           "Expected non-null managed malloc function pointer");

    reinterpret_cast<void *&>(origFreeHost) =
        dlsym(rtLib, ImplT::getPinnedFreeFnName());
    assert(origFreeHost && "Expected non-null Free Pinned Function");

    reinterpret_cast<void *&>(origFreeDevice) =
        dlsym(rtLib, ImplT::getDeviceFreeFnName());
    assert(origFreeDevice && "Expected non-null Device free function pointer");

    reinterpret_cast<void *&>(origRegisterFunction) =
        dlsym(rtLib, ImplT::getUURegisterFunctionFnName());
    assert(origRegisterFunction && "Expected non-null Register Function");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, ImplT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterDeviceVar) =
        dlsym(rtLib, ImplT::getUURegisterVarFnName());
    assert(origRegisterDeviceVar && "Expected non-null register Device Var");

    reinterpret_cast<void *&>(origRegisterFatBinary) =
        dlsym(rtLib, ImplT::getUURegisterFatbinFnName());
    assert(origRegisterFatBinary && "Expected non-null register Device Var");

    if (ImplT::hasFatBinEnd) {
      assert(origRegisterDeviceVar && "Expected non-null register Device Var");
    }
  }
};
//
//  bool RecordKernel(std::string &KernelName, dim3 &gridDim, dim3 blockDim) {
//
//    long largestThreadID = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y)
//    *
//                           (gridDim.z * blockDim.z);
//    auto KD = RecordedDims.find(KernelName);
//    // If we have already recorded this kernel we know this is within our
//    // whitelist
//    if (KD != RecordedDims.end()) {
//      if (KD->second < largestThreadID) {
//        return true;
//      }
//      return false;
//    }
//
//    std::string all("all");
//    auto it = std::find(SymbolWhiteList.begin(), SymbolWhiteList.end(), all);
//    if (it != SymbolWhiteList.end())
//      return true;
//
//    for (auto Name : SymbolWhiteList) {
//      if (KernelName.find(Name) != std::string::npos)
//        return true;
//    }
//
//    return false;
//  }
//
//  void getDeviceInfo(int &WarpSize, int &MultiProcessorCount,
//                     int &MaxGridSizeX) {
//    WarpSize = this->WarpSize;
//    MultiProcessorCount = this->MultiProcessorCount;
//    MaxGridSizeX = this->MaxGridSizeX;
//
//    return;
//  }
//
//  bool isTunable(const void *Func, int &MaxThreadsPerBlock, int Dim) {
//    PREFIX(FuncAttributes) Attr;
//    DeviceRTErrCheck(PREFIX(FuncGetAttributes)(&Attr, Func));
//    DEBUG(printf("deviceFuncGetAttributes sharedSizeBytes %zu numRegs %d "
//                 "maxThreadsPerBlock %d\n",
//                 Attr.sharedSizeBytes, Attr.numRegs,
//                 Attr.maxThreadsPerBlock);)
//    MaxThreadsPerBlock = Attr.maxThreadsPerBlock;
//
//    // TODO: Assumes that using shared memory enforces a hard limit on
//    // dimensions.
//    bool IsSharedZero = (Attr.sharedSizeBytes == 0);
//
//    return IsSharedZero;
//  }
//
//  template <typename dType>
//  HostFuncInfo loadSymbolToHost(void *DevPtr, size_t Bytes) {
//
//    assert(Bytes == sizeof(FuncInfo) && "Func info size do not match");
//    FuncInfo funcData;
//
//    DeviceRTErrCheck(PREFIX(Memcpy)(&funcData, (void *)DevPtr, Bytes,
//                                    PREFIX(MemcpyDeviceToHost)));
//
//    DEBUG(std::cout << "Data Size : " << funcData.elements << " "
//                    << funcData.ptr << "\n";)
//
//    HostFuncInfo hfuncData(funcData);
//    DeviceRTErrCheck(PREFIX(Memcpy)(hfuncData.h_ptr, hfuncData.dev_ptr,
//                                    sizeof(dType) * hfuncData.elements,
//                                    PREFIX(MemcpyDeviceToHost)));
//    return hfuncData;
//  }
//
//  void addGlobal(const std::string &SymbolName, void *DevPtr, size_t Bytes) {
//    if (SymbolName.find("_record_replay_func_info_") != std::string::npos) {
//      auto hfuncData = loadSymbolToHost<uint64_t>(DevPtr, Bytes);
//      auto FuncName = std::regex_replace(
//          SymbolName, std::regex("_record_replay_func_info_"), "");
//
//      DEBUG(std::cout << "Adding: " << FuncName << "\n";)
//      ArgsInfo[FuncName] = hfuncData;
//      ArgsInfo[FuncName].dump(true);
//    } else if (SymbolName.find("_record_replay_descr_") != std::string::npos)
//    {
//      std::string ModuleName = std::regex_replace(
//          SymbolName, std::regex("_record_replay_descr_"), "");
//      DEBUG(std::cout << "Found record replay description\n";)
//      auto llvmIRInfo = loadSymbolToHost<uint8_t>(DevPtr, Bytes);
//      std::error_code EC;
//
//      std::string extracted_ir_fn(
//          Twine(record_replay_dir.string() + ModuleName, ".bc").str());
//      raw_fd_ostream OutBC(extracted_ir_fn, EC);
//      if (EC)
//        throw std::runtime_error("Cannot open device code " +
//        extracted_ir_fn);
//      OutBC << StringRef(reinterpret_cast<const char *>(llvmIRInfo.h_ptr),
//                         llvmIRInfo.elements);
//      DEBUG(std::cout << "Registered Record replay descr";)
//      OutBC.close();
//      llvmIRInfo.dump();
//      DEBUG(std::cout << "One more module file " << extracted_ir_fn << "\n";)
//      ModuleFiles.push_back(std::move(extracted_ir_fn));
//    } else {
//      DEBUG(std::cout << "Device Address of Symbol " << SymbolName << " is "
//                      << DevPtr << " with size: " << Bytes << "\n";)
//
//      TrackedGlobalVars.emplace(
//          SymbolName, std::move(GlobalVar(SymbolName, Bytes, (void
//          *)DevPtr)));
//    }
//
//    // We erase here. This is not a global we would like to track
//  }
//
//  void loadRRGlobals() {
//    static bool RRGlobalsInitialized = false;
//    if (RRGlobalsInitialized)
//      return;
//    for (auto GM = GlobalsMap.begin(); GM != GlobalsMap.end(); GM++) {
//      void *DevPtr;
//      DEBUG(std::cout << "Getting symbol address of symbol " << GM->first
//                      << " with host address: " << (void *)GM->second.second
//                      << "\n";)
//      DeviceRTErrCheck(
//          PREFIX(GetSymbolAddress)(&DevPtr, (const void *)GM->second.second));
//      addGlobal(GM->first, DevPtr, GM->second.first);
//      // hipError_t err = hipGetSymbolAddress(&DevPtr, GM->second.second);
//    }
//    RRGlobalsInitialized = true;
//  }
//
//  const std::filesystem::path &getDataStoreDir() const {
//    return record_replay_dir;
//  }
//
//  uint64_t SuggestedSize() const { return VAMemSuggestedSize; }
//
// public:
//  MemoryManager *MemManager;
//  // Contains name of global and respective size;
//  std::unordered_map<std::string, std::pair<size_t, const char *>> GlobalsMap;
//
//  // All globals we will need to record.
//  std::unordered_map<std::string, GlobalVar> TrackedGlobalVars;
//  std::unordered_map<const void *, std::pair<std::string, std::string>>
//      SymbolTable;
//  std::vector<std::string> SymbolWhiteList;
//  std::vector<std::pair<CudaRegisterFatBinaryArguments *, uint64_t>>
//      FatBinaries;
//
//  std::unordered_map<std::string, HostFuncInfo> ArgsInfo;
//  json::Object RecordedKernels;
//  std::unordered_map<std::string, uint64_t> RecordedDims;
//  json::Array ModuleFiles;
//
// private:
//  void *device_runtime_handle;
//  int WarpSize;
//  int MultiProcessorCount;
//  int MaxGridSizeX;
//  uint64_t VAMemSuggestedSize;
//  std::string record_replay_fn;
//  std::filesystem::path record_replay_dir;
//
//  MnemeRecorder() {
//    MemManager = nullptr;
// #ifdef ENABLE_CUDA
//    device_runtime_handle = dlopen("libcudart.so", RTLD_NOW);
// #else
//    device_runtime_handle = dlopen("libamdhip64.so", RTLD_NOW);
// #endif
//    assert(device_runtime_handle && "Expected non-null");
//
//    // Redirect overloaded device runtime functions.
//    reinterpret_cast<void *&>(deviceLaunchKernelInternal) =
//        dlsym(device_runtime_handle, DEVICE_FUNC("LaunchKernel"));
//    assert(deviceLaunchKernelInternal && "Expected non-null");
//
//    reinterpret_cast<void *&>(deviceMallocInternal) =
//        dlsym(device_runtime_handle, DEVICE_FUNC("Malloc"));
//    assert(deviceMallocInternal && "Expected non-null");
//    reinterpret_cast<void *&>(deviceMallocHostInternal) =
//        dlsym(device_runtime_handle, DEVICE_FUNC("MallocHost"));
//    assert(deviceMallocHostInternal && "Expect/ed non-null");
//    reinterpret_cast<void *&>(deviceMallocManagedInternal) =
//        dlsym(device_runtime_handle, DEVICE_FUNC("MallocManaged"));
//    assert(deviceMallocManagedInternal && "Expected non-null");
//    reinterpret_cast<void *&>(deviceFreeInternal) =
//        dlsym(device_runtime_handle, DEVICE_FUNC("Free"));
//    assert(deviceFreeInternal && "Expected non-null");
//
//    reinterpret_cast<void *&>(__deviceRegisterVarInternal) =
//        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterVar"));
//    assert(__deviceRegisterVarInternal && "Expected non-null");
//
//    reinterpret_cast<void *&>(__deviceRegisterFunctionInternal) =
//        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFunction"));
//    assert(__deviceRegisterFunctionInternal && "Expected non-null");
//    DEBUG(printf("=== Library inited\n");)
//
//    // Gather device info.
//    int DeviceId;
//    DeviceRTErrCheck(PREFIX(GetDevice(&DeviceId)));
//    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&WarpSize,
//    // NOTE: very ugly, thank you vendors!
// #ifdef ENABLE_CUDA
//                                                cudaDevAttrWarpSize,
// #else
//                                                hipDeviceAttributeWarpSize,
// #endif
//                                                DeviceId));
//    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&MultiProcessorCount,
// #ifdef ENABLE_CUDA
//                                                cudaDevAttrMultiProcessorCount,
// #else
//                                                hipDeviceAttributeMultiprocessorCount,
// #endif
//                                                DeviceId));
//    // TODO: We use the x-dimension for now, consider adding y,z if needed.
//    DeviceRTErrCheck(PREFIX(DeviceGetAttribute)(&MaxGridSizeX,
// #ifdef ENABLE_CUDA
//                                                cudaDevAttrMaxGridDimX,
// #else
//                                                hipDeviceAttributeMaxGridDimX,
// #endif
//                                                DeviceId));
//
//    // Gather the symbols whitelist.
//    const char *EnvVarSymbols = std::getenv("RR_SYMBOLS");
//    std::string Symbols = (EnvVarSymbols ? EnvVarSymbols : "all");
//
//    if (!Symbols.empty()) {
//      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
//        endpos = Symbols.find(',', pos);
//        SymbolWhiteList.push_back(Symbols.substr(pos, endpos));
//        pos = endpos + 1;
//      }
//    }
//
//    auto env_rr_file = std::getenv("RR_FILE");
//    if (!env_rr_file) {
//      env_rr_file = (char *)"record_replay.json";
//    }
//
//    record_replay_fn = std::string(env_rr_file);
//
//    auto env_rr_data_directory = std::getenv("RR_DATA_DIR");
//    record_replay_dir =
//        (env_rr_data_directory ? std::string(env_rr_data_directory)
//                               : std::filesystem::current_path().string());
//
//    if (!std::filesystem::is_directory(record_replay_dir)) {
//      throw std::runtime_error("Path :" + record_replay_dir.string() +
//                               " does not exist.\n");
//    }
//    record_replay_dir = std::filesystem::absolute(record_replay_dir);
//
//    const char *EnvVAMemSize = std::getenv("RR_VA_SIZE");
//    VAMemSuggestedSize = ((EnvVAMemSize ? std::atol(EnvVAMemSize) : 12L)) *
//                         1024L * 1024L * 1024L;
//  }
//
//  ~Wrapper() {
//    if (!MemManager)
//      return;
//
//    uintptr_t startAddr = MemManager->StartVAAddr();
//    uint64_t totalSize = MemManager->TotalVASize();
//
//    auto JsonFilename = record_replay_dir / record_replay_fn;
//    std::error_code EC;
//    json::Object record;
//    std::ostringstream oss;
//    oss << std::hex << startAddr;
//
//    record["StartVAAddr"] = json::Value(oss.str());
//    record["TotalSize"] = json::Value(totalSize);
//    record["Kernels"] = json::Value(std::move(RecordedKernels));
//    record["Modules"] = json::Value(
//        std::move(ModuleFiles)); // json::Value(std::move(FuncsInModules));
//    raw_fd_ostream JsonOS(JsonFilename.string(), EC);
//    JsonOS << json::Value(std::move(record));
//    JsonOS.close();
//
//    delete MemManager;
//    MemManager = nullptr;
//  }
//};
} // namespace mneme
