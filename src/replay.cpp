#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "jit.hpp"
#include "macro.hpp"
#include "memory.hpp"

using namespace llvm;
#ifdef ENABLE_CUDA
using namespace jit::cuda;
#endif

class MemoryBlob {
  void *HostPtr;
  void *DevPtr;
  uint64_t MemSize;
  uint64_t MappedMemSize;
  bool Mapped;
  bool Copied;
  CUmemGenericAllocationHandle MemHandle;

public:
  MemoryBlob() : HostPtr(nullptr), DevPtr(nullptr), MemSize(0), Mapped(false) {}
  MemoryBlob(void *hPtr, uint64_t size)
      : DevPtr(nullptr), MemSize(size), Mapped(false), Copied(false) {
    HostPtr = new uint8_t[MemSize];
    std::memcpy(HostPtr, hPtr, size);
  }
  MemoryBlob(MemoryBlob &&orig)
      : HostPtr(orig.HostPtr), DevPtr(orig.DevPtr), MemSize(orig.MemSize),
        Mapped(orig.Mapped), Copied(orig.Copied) {
    orig.HostPtr = nullptr;
    orig.DevPtr = nullptr;
  }
  MemoryBlob(const MemoryBlob &) = delete;
  MemoryBlob &operator=(const MemoryBlob &) = delete;

  void dump() const {
    std::cout << "MemSize: " << MemSize << " Mapped: " << Mapped
              << " Copied: " << Copied << "\n";
  }

  void AllocateDevice() {
    cudaErrCheck(cudaMalloc(&DevPtr, MemSize));
    Mapped = false;
  }

  void MemMapToDevice(void *req_addr) {
    // TODO: We currently ignore the address, this can result in weird issues
    // later on
    MemHandle = memory::cuda::MemMapToDevice(&DevPtr, req_addr, MemSize,
                                             MappedMemSize, 0);
    std::cout << "SUCCESS:: " << DevPtr << " Req Size: " << MemSize
              << " Provided : " << MappedMemSize << "\n";
    if (req_addr != DevPtr) {
      std::cerr << "Requested " << req_addr << " But got " << DevPtr << "\n";
      report_fatal_error("Error Address does not match record run\n");
    }
    Mapped = true;
    return;
  }

  void copyToDevice() {
    cudaErrCheck(cudaMemcpy(DevPtr, HostPtr, MemSize, cudaMemcpyHostToDevice));
    Copied = true;
  }

  void copyFromDevice() {
    cudaErrCheck(cudaMemcpy(HostPtr, DevPtr, MemSize, cudaMemcpyDeviceToHost));
  }

  bool compare(MemoryBlob &blob) {
    if (MemSize != blob.MemSize)
      return false;

    uint8_t *ptr1 = (uint8_t *)HostPtr;
    uint8_t *ptr2 = (uint8_t *)blob.HostPtr;
    return std::memcmp(ptr1, ptr2, MemSize) == 0;
  }

  ~MemoryBlob() {
    if (HostPtr)
      delete[] (uint8_t *)HostPtr;
    if (DevPtr) {
      if (!Mapped) {
        cudaErrCheck(cudaFree(DevPtr));
      } else {
        memory::cuda::MemoryUnMap(DevPtr, MemHandle, MappedMemSize);
      }
    }
  }
};

struct BinarySnapshot {
  uint8_t **args;
  uint64_t num_args;
  std::vector<uint64_t> arg_size;
  std::unordered_map<intptr_t, MemoryBlob> DeviceBlobs;
  std::unordered_map<std::string, GlobalVar> TrackedGlobalVars;

private:
  BinarySnapshot() : args(nullptr), num_args(0), arg_size(0) {}
  BinarySnapshot(uint8_t **args, int64_t nargs, std::vector<uint64_t> &sizes)
      : args(args), num_args(nargs), arg_size(sizes) {}

public:
  static void **allocateArgs(std::vector<uint64_t> &args) {
    void **ptrs = new void *[args.size()];
    for (int i = 0; i < args.size(); i++) {
      ptrs[i] = new uint8_t[args[i]];
    }
    return ptrs;
  }

  static BinarySnapshot Load(std::string FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Snapshot =
        MemoryBuffer::getFile(FileName, /* isText */ false,
                              /* RequiresNullTerminator */ false);

    // read argument values
    const void *BufferPtr = Snapshot.get()->getBufferStart();
    const void *StartPtr = BufferPtr;
    uint64_t num_args = (*(uint64_t *)BufferPtr);
    BufferPtr = advanceVoidPtr(BufferPtr, sizeof(uint64_t));
    std::cout << "Read " << num_args << " arguments\n";
    std::vector<uint64_t> arg_sizes;
    for (int i = 0; i < num_args; i++) {
      arg_sizes.push_back(*(uint64_t *)BufferPtr);
      BufferPtr = advanceVoidPtr(BufferPtr, sizeof(uint64_t));
    }

    void **args = allocateArgs(arg_sizes);

    for (int i = 0; i < num_args; i++) {
      std::memcpy(args[i], BufferPtr, arg_sizes[i]);
      BufferPtr = advanceVoidPtr(BufferPtr, arg_sizes[i]);
    }
    // Header of arguments Finished
    BinarySnapshot MemState((uint8_t **)args, num_args, arg_sizes);

    size_t num_globals = *(size_t *)BufferPtr;
    BufferPtr = advanceVoidPtr(BufferPtr, sizeof(size_t));

    for (int i = 0; i < num_globals; i++) {
      GlobalVar global_variable(&BufferPtr);
      MemState.TrackedGlobalVars.emplace(global_variable.Name,
                                         std::move(global_variable));
      std::cout << "Total Bytes read: "
                << (uintptr_t)BufferPtr - (uintptr_t)StartPtr << "\n";
    }

    for (auto &GV : MemState.TrackedGlobalVars) {
      GV.second.dump();
    }

    // Read till the end of the file
    while (BufferPtr != Snapshot.get()->getBufferEnd()) {
      intptr_t dPtr = *(intptr_t *)BufferPtr;
      BufferPtr = advanceVoidPtr(BufferPtr, sizeof(intptr_t));

      size_t memBlobSize = *(uint64_t *)BufferPtr;
      BufferPtr = advanceVoidPtr(BufferPtr, sizeof(uint64_t));

      MemState.DeviceBlobs.emplace(dPtr,
                                   MemoryBlob((void *)BufferPtr, memBlobSize));
      std::cout << "I just added :\n";
      MemState.DeviceBlobs[dPtr].dump();
      BufferPtr = advanceVoidPtr(BufferPtr, memBlobSize);
    }

    return MemState;
  }

  void dump() const {
    std::cout << "Snapshot has " << num_args;
    std::cout << " Arguments:\n";
    for (auto I : arg_size) {
      std::cout << "Argument Size: " << I << "\n";
    }

    for (int i = 0; i < num_args; i++) {
      std::cout << "Argument :" << i
                << " Addr:" << (void *)(*(uint64_t *)args[i]) << "\n";
    }

    for (auto &KV : DeviceBlobs) {
      std::cout << "Information About Device Memory at Address: "
                << (void *)KV.first << "\n";
      KV.second.dump();
    }
  }

  void AllocateDevice(bool Map = true) {
    if (Map) {
      for (auto &KV : DeviceBlobs) {
        KV.second.MemMapToDevice((void *)KV.first);
      }
    } else {
      for (auto &KV : DeviceBlobs) {
        KV.second.AllocateDevice();
      }
    }
  }

  void copyToDevice() {
    for (auto &KV : DeviceBlobs) {
      KV.second.copyToDevice();
    }
    for (auto &KV : TrackedGlobalVars) {
      std::cout << "Copying to global: " << KV.first << "\n";
      KV.second.copyToDevice();
    }
  }

  void LoadGlobals(CUmodule &CUMod) {
    for (auto &KV : TrackedGlobalVars) {
      KV.second.setDevPtrFromModule(CUMod);
    }
  }

  void copyFromDevice() {
    for (auto &KV : DeviceBlobs) {
      KV.second.copyFromDevice();
    }
    for (auto &KV : TrackedGlobalVars) {
      std::cout << "Copying back global: " << KV.first << "\n";
      KV.second.copyFromDevice();
    }
  }

  /* Returns true when blobs are the same */
  bool compare(BinarySnapshot &other) {
    for (auto &KV : DeviceBlobs) {
      uint64_t devAddr = KV.first;
      auto otherEntry = other.DeviceBlobs.find(devAddr);
      if (otherEntry == other.DeviceBlobs.end()) {
        std::cout << "Cannot find dev addr \n";
        return false;
      }

      if (!KV.second.compare(otherEntry->second)) {
        std::cout << "Dev Memory differs \n";
        return false;
      }
    }

    for (auto &KV : TrackedGlobalVars) {
      auto otherEntry = other.TrackedGlobalVars.find(KV.first);
      if (otherEntry == other.TrackedGlobalVars.end()) {
        std::cout << "Cannot find the same symbol, globals differ \n";
        return false;
      }
      if (!KV.second.compare(otherEntry->second)) {
        std::cout << "Globals differ\n";
        return false;
      }
    }
    return true;
  }
};

std::pair<std::string, CUdevice> init() {
  CUdevice CUDev;
  CUcontext CUCtx;

  cuErrCheck(cuInit(0));

  CUresult CURes = cuCtxGetDevice(&CUDev);
  if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
    // TODO: is selecting device 0 correct?
    cuErrCheck(cuDeviceGet(&CUDev, 0));

  cuErrCheck(cuCtxGetCurrent(&CUCtx));
  if (!CUCtx)
    cuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));

  int CCMajor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  int CCMinor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  std::string CudaArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);
  return std::make_pair(CudaArch, CUDev);
}

dim3 getDim3(json::Object &Info, std::string key) {
  auto JObject = Info.getObject(key);
  long x = *JObject->getInteger("x");
  long y = *JObject->getInteger("y");
  long z = *JObject->getInteger("z");
  return dim3(x, y, z);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Wrong CLI, expecting:" << argv[0]
              << " 'Path to JSON file' 'Function Name to Replay";
    exit(-1);
  }

  auto DeviceSpec = init();
  std::string DeviceArch = DeviceSpec.first;

  std::string JSONFileName(argv[1]);
  std::string KernelName(argv[2]);

  // Load JSON File
  ErrorOr<std::unique_ptr<MemoryBuffer>> KernelInfoMB =
      MemoryBuffer::getFile(JSONFileName, /* isText */ true,
                            /* RequiresNullTerminator */ true);

  Expected<json::Value> JsonInfo = json::parse(KernelInfoMB.get()->getBuffer());
  if (auto Err = JsonInfo.takeError())
    report_fatal_error("Cannot parse the kernel info json file");

  json::Object *RecordInfo = JsonInfo->getAsObject()->getObject("kernels");
  json::Value *Info = RecordInfo->get(KernelName);
  if (Info == nullptr) {
    report_fatal_error("Requested function does not have a record entry");
  }
  json::Object KernelInfo = *Info->getAsObject();

  std::string iDeviceMemory(KernelInfo.getString("InputData")->str());
  std::string oDeviceMemory(KernelInfo.getString("OutputData")->str());

  dim3 gridDim = getDim3(KernelInfo, "Grid");
  dim3 blockDim = getDim3(KernelInfo, "Block");
  size_t SharedMem = *KernelInfo.getInteger("SharedMemory");

  // Load Kernel Input
  BinarySnapshot input = BinarySnapshot::Load(iDeviceMemory);
  BinarySnapshot output = BinarySnapshot::Load(oDeviceMemory);
  input.AllocateDevice(true);

  json::Object *RecordedModules = JsonInfo->getAsObject()->getObject("modules");
  assert(RecordedModules->size() == 1 &&
         "Record replay does not support multiple modules");

  std::unordered_map<std::string, std::vector<std::string>> IRtoFuncMap;

  for (auto KV : *RecordedModules) {
    std::string LLVMIR = KV.first.str();
    json::Array &FuncNames = *KV.second.getAsArray();
    std::vector<std::string> FNNames;
    for (auto A : FuncNames) {
      FNNames.push_back(A.getAsString()->str());
    }
    IRtoFuncMap[LLVMIR] = std::move(FNNames);
  }

  auto Entry = *IRtoFuncMap.begin();
  std::string IRFn = Twine(Entry.first, ".bc").str();

  // Load IR
  InitJITEngine();
  auto TIR = loadIR(IRFn);
  if (auto E = TIR.takeError())
    report_fatal_error("Could Not load llvm IR");

  auto &M = *TIR.get().getModuleUnlocked();

  OptimizeIR(M, 3, DeviceArch);

  SmallVector<char, 4096> PTXStr;
  IRToBackEnd(M, DeviceArch, PTXStr);
  StringRef PTX(PTXStr.data(), PTXStr.size());

#ifdef __ENABLE_DEBUG__
  std::error_code EC;
  std::string rrBC("ptx-ir.s");
  raw_fd_ostream OutBC(rrBC, EC);
  if (EC)
    throw std::runtime_error("Cannot open device code " + rrBC);
  OutBC << PTX.str();
  OutBC.close();
#endif

  // Lower to device and get function
  CUmodule DevModule;
  CUfunction DevFunction;
  CreateDeviceObject(PTX, DevModule);
  GetDeviceFunction(DevFunction, DevModule, KernelName);
  input.LoadGlobals(DevModule);
  input.copyToDevice();
  input.dump();

  // Execute Device
  LaunchFunction(DevModule, DevFunction, gridDim, blockDim, SharedMem,
                 (void **)input.args);
  std::cout << "Done\n";

  // Compare "input memory with output memory"

  input.copyFromDevice();

  if (input.compare(output)) {
    std::cout << "Results match [Verified]";
    return 0;
  }
  std::cout << "Results do not match [incorrect replay]";
  return 1;
}
