#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <llvm/IR/Constants.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "jit.hpp"
#include "macro.hpp"
#include "memory.hpp"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>

using namespace llvm;
#ifdef ENABLE_CUDA
using namespace jit::cuda;
#endif

static cl::OptionCategory ReplayCategory("Replay Tool Options",
                                         "Record Replay CLI options.");

static cl::opt<long>
    blockDimx("blockDimx",
              cl::desc("The number of threads on x dimension to "
                       "assign to a block during kernel replay"),
              cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<long>
    blockDimy("blockDimy",
              cl::desc("The number of threads on y dimension to "
                       "assign to a block during kernel replay"),
              cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<long>
    blockDimz("blockDimz",
              cl::desc("The number of threads on z dimension to "
                       "assign to a block during kernel replay"),
              cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<long>
    gridDimx("gridDimx",
             cl::desc("The number of threads on x dimension to "
                      "assign to a grid during kernel replay"),
             cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<long>
    gridDimy("gridDimy",
             cl::desc("The number of threads on y dimension to "
                      "assign to a grid during kernel replay"),
             cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<long>
    gridDimz("gridDimz",
             cl::desc("The number of threads on z dimension to "
                      "assign to a grid during kernel replay"),
             cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<std::string>
    KernelName("kernel-name", cl::desc("The kernel function to replay"),
               cl::Required, cl::cat(ReplayCategory));

static cl::opt<int> MaxThreads("max-threads",
                               cl::desc("Passing MaxThreads Launchbound"),
                               cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<int> MinBlocks("min-blocks",
                              cl::desc("Passing MinBlocks Launchbound"),
                              cl::init(-1), cl::cat(ReplayCategory));

static cl::opt<bool> SaveTemps("save-temps",
                               cl::desc("Save temporal files to disk"),
                               cl::init(false), cl::cat(ReplayCategory));

static cl::opt<int> Repeat("repeat", cl::desc("Number of repeat runs"),
                           cl::init(1), cl::cat(ReplayCategory));

static cl::opt<std::string> RRJson(
    "record-replay-json",
    cl::desc("The json file containing metadata for all recorded kernels"),
    cl::Required);

void WriteToFile(Module &M, std::string FileName, bool save = false) {
  if (!save)
    return;
  std::error_code EC;
  raw_fd_ostream OutBC(FileName, EC);
  if (EC)
    throw std::runtime_error("Cannot write code " + FileName);
  OutBC << M;
  OutBC.close();
}

void AssignLaunchBounds(llvm::Module &M, llvm::Function &F, int MaxThreads,
                        int MinBlocks) {

  llvm::NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");
  if (MaxThreads != -1) {
    llvm::Metadata *MDVals[] = {
        llvm::ConstantAsMetadata::get(&F),
        llvm::MDString::get(M.getContext(), "maxntidx"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(M.getContext()), MaxThreads))};
    MD->addOperand(llvm::MDNode::get(M.getContext(), MDVals));
  }

  if (MinBlocks != -1) {
    llvm::Metadata *MDVals[] = {
        llvm::ConstantAsMetadata::get(&F),
        llvm::MDString::get(M.getContext(), "minctasm"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(M.getContext()), MinBlocks))};
    MD->addOperand(llvm::MDNode::get(M.getContext(), MDVals));
  }
}

class MemoryBlob {
  MemoryManager &MemManager;
  void *HostPtr;
  void *DevPtr;
  uint64_t MemSize;
  uint64_t MappedMemSize;
  bool Mapped;
  bool Copied;

public:
  MemoryBlob(MemoryManager &MemManager, void *hPtr, uint64_t size)
      : MemManager(MemManager), DevPtr(nullptr), MemSize(size), Mapped(false),
        Copied(false) {
    HostPtr = new uint8_t[MemSize];
    std::memcpy(HostPtr, hPtr, size);
  }
  MemoryBlob(MemoryBlob &&orig)
      : MemManager(orig.MemManager), HostPtr(orig.HostPtr), DevPtr(orig.DevPtr),
        MemSize(orig.MemSize), Mapped(orig.Mapped), Copied(orig.Copied) {
    orig.HostPtr = nullptr;
    orig.DevPtr = nullptr;
  }

  MemoryBlob(const MemoryBlob &) = delete;
  MemoryBlob &operator=(const MemoryBlob &) = delete;

  void dump() const {
    DEBUG(std::cout << "MemSize: " << MemSize << " Mapped: " << Mapped
                    << " Copied: " << Copied << "\n";)
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

  void Reserve(void *DevAddr = nullptr) {
    DevPtr = MemManager.allocate(MemSize, DevAddr);
  }

  ~MemoryBlob() {
    if (HostPtr)
      delete[] (uint8_t *)HostPtr;
    if (DevPtr)
      MemManager.release(DevPtr);
  }
};

struct BinarySnapshot {
  MemoryManager &MemManager;
  uint8_t **args;
  uint64_t num_args;
  std::vector<uint64_t> arg_size;
  std::map<intptr_t, MemoryBlob> DeviceBlobs;
  std::unordered_map<std::string, GlobalVar> TrackedGlobalVars;

private:
  BinarySnapshot(MemoryManager &MemManager, uint8_t **args, int64_t nargs,
                 std::vector<uint64_t> &sizes)
      : MemManager(MemManager), args(args), num_args(nargs), arg_size(sizes) {}

public:
  static void **allocateArgs(std::vector<uint64_t> &args) {
    void **ptrs = new void *[args.size()];
    for (int i = 0; i < args.size(); i++) {
      ptrs[i] = new uint8_t[args[i]];
    }
    return ptrs;
  }

  static BinarySnapshot Load(MemoryManager &MemManager, std::string FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Snapshot =
        MemoryBuffer::getFile(FileName, /* isText */ false,
                              /* RequiresNullTerminator */ false);

    // read argument values
    const void *BufferPtr = Snapshot.get()->getBufferStart();
    const void *StartPtr = BufferPtr;
    uint64_t num_args = (*(uint64_t *)BufferPtr);
    BufferPtr = util::advanceVoidPtr(BufferPtr, sizeof(uint64_t));
    DEBUG(std::cout << "Read " << num_args << " arguments\n";)
    std::vector<uint64_t> arg_sizes;
    for (int i = 0; i < num_args; i++) {
      arg_sizes.push_back(*(uint64_t *)BufferPtr);
      BufferPtr = util::advanceVoidPtr(BufferPtr, sizeof(uint64_t));
    }

    void **args = allocateArgs(arg_sizes);

    for (int i = 0; i < num_args; i++) {
      std::memcpy(args[i], BufferPtr, arg_sizes[i]);
      BufferPtr = util::advanceVoidPtr(BufferPtr, arg_sizes[i]);
    }
    // Header of arguments Finished
    BinarySnapshot MemState(MemManager, (uint8_t **)args, num_args, arg_sizes);

    size_t num_globals = *(size_t *)BufferPtr;
    BufferPtr = util::advanceVoidPtr(BufferPtr, sizeof(size_t));

    for (int i = 0; i < num_globals; i++) {
      GlobalVar global_variable(&BufferPtr);
      MemState.TrackedGlobalVars.emplace(global_variable.Name,
                                         std::move(global_variable));
      DEBUG(std::cout << "Total Bytes read: "
                      << (uintptr_t)BufferPtr - (uintptr_t)StartPtr << "\n";)
    }

    for (auto &GV : MemState.TrackedGlobalVars) {
      GV.second.dump();
    }

    // Read till the end of the file
    while (BufferPtr != Snapshot.get()->getBufferEnd()) {
      intptr_t dPtr = *(intptr_t *)BufferPtr;
      BufferPtr = util::advanceVoidPtr(BufferPtr, sizeof(intptr_t));

      size_t memBlobSize = *(uint64_t *)BufferPtr;
      BufferPtr = util::advanceVoidPtr(BufferPtr, sizeof(uint64_t));

      MemState.DeviceBlobs.emplace(
          dPtr, MemoryBlob(MemManager, (void *)BufferPtr, memBlobSize));
      auto it = MemState.DeviceBlobs.find(dPtr);
      if (it != MemState.DeviceBlobs.end()) {
        it->second.dump();
      }

      BufferPtr = util::advanceVoidPtr(BufferPtr, memBlobSize);
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
        std::cout << "Starting with first addr " << (void *)KV.first << "\n";
        KV.second.Reserve((void *)KV.first);
      }
    } else {
      for (auto &KV : DeviceBlobs) {
        KV.second.Reserve();
      }
    }
  }

  void copyToDevice() {
    for (auto &KV : DeviceBlobs) {
      KV.second.copyToDevice();
    }
    for (auto &KV : TrackedGlobalVars) {
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

dim3 getDim3(json::Object &Info, std::string key, long opt_x, long opt_y,
             long opt_z) {
  auto JObject = Info.getObject(key);
  long x = (opt_x == -1) ? *JObject->getInteger("x") : opt_x;
  long y = (opt_y == -1) ? *JObject->getInteger("y") : opt_y;
  long z = (opt_z == -1) ? *JObject->getInteger("z") : opt_z;

  std::cout << "Recorded " << key << " configuration: ("
            << *JObject->getInteger("x") << ", " << *JObject->getInteger("y")
            << ", " << *JObject->getInteger("z") << ")\n";
  std::cout << "Replay " << key << " configuration: (" << x << ", " << y << " ,"
            << z << ")\n";

  return dim3(x, y, z);
}

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(ReplayCategory);
  cl::ParseCommandLineOptions(argc, argv, "GPU Replay Tool\n");

  auto DeviceSpec = init();
  std::string DeviceArch = DeviceSpec.first;

  if (!std::filesystem::exists(RRJson.getValue())) {
    std::cerr << "Record-Replay JSON File " << RRJson << " does not exist!"
              << std::endl;
    return 1;
  }
  // Load JSON File
  ErrorOr<std::unique_ptr<MemoryBuffer>> KernelInfoMB =
      MemoryBuffer::getFile(RRJson, /* isText */ true,
                            /* RequiresNullTerminator */ true);

  Expected<json::Value> JsonInfo = json::parse(KernelInfoMB.get()->getBuffer());
  if (auto Err = JsonInfo.takeError())
    report_fatal_error("Cannot parse the kernel info json file");

  auto JSONRoot = JsonInfo->getAsObject();
  json::Value *VAAddrValue = JSONRoot->get("StartVAAddr");
  if (VAAddrValue == nullptr) {
    std::cerr << "JSON file does not contain Starting VA field (StartVAAddr)\n";
    return 1;
  }

  std::string VAAddrStr = VAAddrValue->getAsString()->str();
  uintptr_t StartVAAddr = std::stoull(VAAddrStr, nullptr, 16);
  json::Value *TotalSizeValue = JSONRoot->get("TotalSize");
  if (TotalSizeValue == nullptr) {
    std::cerr << "JSON file does not contain Total VA Size (TotalSize) field\n";
    return 1;
  }

  auto TotalSize = TotalSizeValue->getAsUINT64();
  if (!TotalSize)
    report_fatal_error("Cannot read TotalSize value");

  uint64_t VASize = *TotalSize;

  MemoryManager MemManager(VASize, (void *)StartVAAddr);

  if (MemManager.StartVAAddr() != StartVAAddr)
    report_fatal_error("Could not reserve the requested address range");

  json::Object *RecordInfo = JSONRoot->getObject("Kernels");
  json::Value *Info = RecordInfo->get(KernelName);
  if (Info == nullptr) {
    report_fatal_error("Requested function does not have a record entry");
  }
  json::Object KernelInfo = *Info->getAsObject();

  std::string iDeviceMemory(KernelInfo.getString("InputData")->str());
  if (!std::filesystem::exists(iDeviceMemory)) {
    std::cerr << "Record-Replay snapshot File " << iDeviceMemory
              << " does not exist!" << std::endl;
    return 1;
  }

  std::string oDeviceMemory(KernelInfo.getString("OutputData")->str());
  if (!std::filesystem::exists(oDeviceMemory)) {
    std::cerr << "Record-Replay snapshot File " << oDeviceMemory
              << " does not exist!" << std::endl;
    return 1;
  }

  dim3 gridDim = getDim3(KernelInfo, "Grid", gridDimx.getValue(),
                         gridDimy.getValue(), gridDimz.getValue());
  dim3 blockDim = getDim3(KernelInfo, "Block", blockDimx, blockDimy, blockDimz);
  size_t SharedMem = *KernelInfo.getInteger("SharedMemory");

  // Load Kernel Input
  BinarySnapshot input = BinarySnapshot::Load(MemManager, iDeviceMemory);
  BinarySnapshot output = BinarySnapshot::Load(MemManager, oDeviceMemory);
  input.AllocateDevice(true);

  json::Array *RecordedModules = JsonInfo->getAsObject()->getArray("Modules");
  assert(RecordedModules->size() == 1 &&
         "Record replay does not support multiple modules");
  auto moduleFN = RecordedModules->front().getAsString();
  if (!moduleFN.has_value()) {
    throw std::runtime_error(
        "Expecting at least a single module file name, got None");
  }
  std::string IRFn(moduleFN.value().str());

  // Load IR
  InitJITEngine();

  auto TIR = loadIR(IRFn);
  if (auto E = TIR.takeError())
    report_fatal_error("Could Not load llvm IR");

  auto &M = *TIR.get().getModuleUnlocked();

  OptimizeIR(M, 3, DeviceArch);

  Function *KernelFunc = M.getFunction(KernelName);
  AssignLaunchBounds(M, *KernelFunc, MaxThreads, MinBlocks);
  WriteToFile(M, KernelName + ".after_bounds.bc", SaveTemps);

  SmallVector<char, 4096> PTXStr;
  IRToBackEnd(M, DeviceArch, PTXStr);
  StringRef PTX(PTXStr.data(), PTXStr.size());

#ifdef ENABLE_DEBUG
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
  DEBUG(input.dump());
  std::vector<float> timings;
  for (int i = 0; i < Repeat; i++) {
    input.LoadGlobals(DevModule);
    input.copyToDevice();
    float milliseconds =
        LaunchFunction(DevModule, DevFunction, gridDim, blockDim, SharedMem,
                       (void **)input.args);
    timings.push_back(milliseconds);
  }

  float sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  float mean = sum / timings.size();

  // Compute the standard deviation
  float sq_sum =
      std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
  float std_dev = std::sqrt(sq_sum / timings.size() - mean * mean);
  std::cout << "Mean execution time (ms): " << mean
            << " standard Deviation: " << std_dev << " Block Dims ("
            << blockDim.x << "," << blockDim.y << "," << blockDim.z << ")"
            << " Grid Dims (" << gridDim.x << "," << gridDim.y << ","
            << gridDim.z << ") MaxThreads:" << MaxThreads << " MinBlocks "
            << MinBlocks << "\n";

  std::cout << "Done\n";
  input.copyFromDevice();

  if (input.compare(output)) {
    std::cout << "Results match [Verified]\n";
    return 0;
  }
  std::cout << "Results do not match [incorrect replay]";
  return 1;
}
