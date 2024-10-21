#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>

#include <iostream>
#include <memory>
#include <string>

#include "device_types.hpp"
#include "jit.hpp"
#include "macro.hpp"

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#elif defined(ENABLE_HIP)
#include <hip/hiprtc.h>
#endif

using namespace llvm;

#define hiprtcErrCheck(CALL)                                                   \
  {                                                                            \
    hiprtcResult err = CALL;                                                   \
    if (err != HIPRTC_SUCCESS) {                                               \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hiprtcGetErrorString(err));                                       \
      abort();                                                                 \
    }                                                                          \
  }

static codegen::RegisterCodeGenFlags CFG;

static Expected<std::unique_ptr<TargetMachine>> createTargetMachine(
    Module &M, StringRef CPU,
    llvm::CodeGenOptLevel CGOptLevel = llvm::CodeGenOptLevel::Aggressive) {
  Triple TT(M.getTargetTriple());

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TT);

  std::optional<Reloc::Model> RelocModel;
  if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel = M.getCodeModel();

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TT);

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features.getString(),
                             Options, RelocModel, CodeModel, CGOptLevel));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

namespace cuda {

static void setBound(Module *M, Function *F, const char *handler, int value) {
  NamedMDNode *NvvmAnnotations = M->getNamedMetadata("nvvm.annotations");
  assert(NvvmAnnotations && "Expected non-null nvvm.annotations metadata");
  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F),
      llvm::MDString::get(M->getContext(), handler),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(M->getContext()), value))};
  NvvmAnnotations->addOperand(llvm::MDNode::get(M->getContext(), MDVals));
}

void SetLaunchBounds(Module &M, std::string &FnName, int MaxThreads,
                     int MinBlocks) {

  Function *F = M.getFunction(FnName);
  if (MaxThreads != -1) {
    MaxThreads = std::min(1024, MaxThreads);
    setBound(&M, F, "maxntidx", MaxThreads);
  }

  if (MaxThreads != -1) {
    setBound(&M, F, "minctasm", MinBlocks);
  }
}

static void codegen(Module &M, StringRef Arch, SmallVectorImpl<char> &Str) {
  auto TMExpected = createTargetMachine(M, Arch);
  if (!TMExpected)
    FATAL_ERROR(toString(TMExpected.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  raw_svector_ostream PTXOS(Str);
  TM->addPassesToEmitFile(PM, PTXOS, nullptr,
                          llvm::CodeGenFileType::AssemblyFile,
                          /* DisableVerify */ false, MMIWP);

  PM.run(M);

  Str.push_back('\0');
}

#ifdef ENABLE_CUDA
void BackEndToDeviceMod(llvm::StringRef &CodeRepr, gpu::DeviceModule &Mod) {
  char out[1024 * 1024];
  char err[1024 * 1024];
  CUjit_option array[5] = {CU_JIT_INFO_LOG_BUFFER,
                           CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                           CU_JIT_ERROR_LOG_BUFFER, CU_JIT_LOG_VERBOSE,
                           CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  int64_t values[5] = {(int64_t)out, 1024 * 1024, (int64_t)err, 1, 1024 * 1024};
  int numOptions = 0;
#ifdef __ENABLE_DEBUG__
  numOptions = 5;
#endif // __ENABLE_DEBUG__
  // TODO: This is not good for debug runs, cuErrCheck will terminate in case of
  // an error before we output the logged errors.
  cuErrCheck(
      cuModuleLoadDataEx(&CUMod, PTX.data(), 0, array, (void **)&values));

#ifdef __ENABLE_DEBUG__
  std::cout << "STDOUT\n";
  std::cout << out << "\n";
  std::cout << "STDERR\n";
  std::cout << err << "\n";
#endif // __ENABLE_DEBUG__
  return;
}
#endif

} // namespace cuda

namespace hip {

void SetLaunchBounds(Module &M, std::string &FnName, int MaxThreads,
                     int MinBlocks) {

  Function *F = M.getFunction(FnName);
  F->addFnAttr("amdgpu-flat-work-group-size",
               "1," + std::to_string(std::min(1024, MaxThreads)));
}

static void codegen(Module &M, StringRef Arch, SmallVectorImpl<char> &Str) {
  raw_svector_ostream ModuleBufferOS(Str);
  WriteBitcodeToFile(M, ModuleBufferOS);
}

#ifdef ENABLE_HIP
void BackEndToDeviceMod(llvm::StringRef &CodeRepr, gpu::DeviceModule &Mod) {
  hiprtcLinkState hip_link_state_ptr;

  // TODO: Dynamic linking is to be supported through hiprtc. Currently
  // the interface is limited and lacks support for linking globals.
  // Indicative code here is for future re-visit.
  const char *OptArgs[] = {};
  //    const char *OptArgs[] = {"-mllvm", "-amdgpu-internalize-symbols",
  //    "-mllvm",
  //                             "-unroll-threshold=1000", "-march=gfx90a"};
  std::vector<hiprtcJIT_option> JITOptions = {
      HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t OptArgsSize = 0;
  const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
  hiprtcErrCheck(hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                                  (void **)JITOptionsValues,
                                  &hip_link_state_ptr));

  hiprtcErrCheck(hiprtcLinkAddData(
      hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
      (void *)CodeRepr.data(), CodeRepr.size(), "", 0, nullptr, nullptr));

  char *BinOut;
  size_t BinSize;
  hiprtcErrCheck(
      hiprtcLinkComplete(hip_link_state_ptr, (void **)&BinOut, &BinSize));

  DeviceRTErrCheck(hipModuleLoadData(&Mod, BinOut));
}
#endif

} // namespace hip

static inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}

static void runOptimizationPassPipeline(Module &M, StringRef Target,
                                        const OptimizationLevel &lvl) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Target);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager Passes = PB.buildPerModuleDefaultPipeline(lvl);
  Passes.run(M, MAM);
}

void InitJITEngine() {
#ifdef ENABLE_CUDA
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#elif defined(ENABLE_HIP)
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
#endif
}

std::string getArch() {
  gpu::DeviceHandle Dev;
  gpu::DeviceContext Ctx;

#ifdef ENABLE_CUDA
  cuErrCheck(DRIVER_PREFIX(Init)(0));
  gpu::DeviceResult Res = cuCtxGetDevice(&Dev);
  if (Res == CUDA_ERROR_INVALID_CONTEXT or !Dev)
    // TODO: is selecting device 0 correct?
    cuErrCheck(cuDeviceGet(&CUDev, 0));
  cuErrCheck(cuCtxGetCurrent(&Ctx));
  if (!Ctx)
    cuErrCheck(cuCtxCreate(&Ctx, 0, CUDev));

  int CCMajor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  int CCMinor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  std::string CudaArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);
  return CudaArch;

#elif defined(ENABLE_HIP)
  DeviceRTErrCheck(DRIVER_PREFIX(Init)(0));
  DeviceRTErrCheck(hipGetDevice(&Dev));
  hipDeviceProp_t device_prop;

  // Get properties of the current device
  hipGetDeviceProperties(&device_prop, Dev);

  // Get the full architecture name (e.g., gfx90a:sramecc+:xnack-)
  std::string arch_name = device_prop.gcnArchName;

  // Find the colon (:) to isolate the base architecture
  std::string HipArch = arch_name.substr(0, arch_name.find(':'));

  // Print the base architecture

  // Print the architecture name
  return std::string(HipArch);
#endif
}

Expected<llvm::orc::ThreadSafeModule> loadIR(StringRef FileName) {
  SMDiagnostic Err;
  auto Ctx = std::make_unique<LLVMContext>();
  if (auto M = parseIRFile(FileName, Err, *Ctx)) {
    return llvm::orc::ThreadSafeModule(std::move(M), std::move(Ctx));
  }
  return createSMDiagnosticError(Err);
}

void OptimizeIR(Module &M, unsigned int lvl, std::string &CudaArch) {
  // TODO: Specialize based on lvl
  runOptimizationPassPipeline(M, CudaArch, OptimizationLevel::O3);
}

void IRToBackEnd(Module &M, std::string &DeviceArch,
                 SmallVectorImpl<char> &CodeRepr) {
  gpu::codegen(M, DeviceArch, CodeRepr);
}

void CreateDeviceObject(StringRef &CodeRepr, gpu::DeviceModule &Mod) {
  gpu::BackEndToDeviceMod(CodeRepr, Mod);
}

void SetLaunchBounds(Module &M, std::string &FnName, int MaxThreads,
                     int MinBlocks) {
  gpu::SetLaunchBounds(M, FnName, MaxThreads, MinBlocks);
}

void GetDeviceFunction(gpu::DeviceFunction &Func, gpu::DeviceModule &Mod,
                       StringRef FunctionName) {
#ifdef ENABLE_CUDA
  cuErrCheck(cuModuleGetFunction(&Func, Mod, FunctionName.data()));
#elif defined(ENABLE_HIP)
  DeviceRTErrCheck(hipModuleGetFunction(&Func, Mod, FunctionName.data()));
#endif
}

float LaunchFunction(gpu::DeviceModule &Mod, gpu::DeviceFunction &Func,
                     dim3 GridDim, dim3 &BlockDim, uint64_t ShMemSize,
                     void **KernelArgs) {
  float milliseconds = 0;
  gpu::DeviceEvent start, stop;
  DeviceRTErrCheck(PREFIX(EventCreate)(&start));
  DeviceRTErrCheck(PREFIX(EventCreate)(&stop));

  DeviceRTErrCheck(PREFIX(EventRecord)(start));
#ifdef ENABLE_CUDA
  cuLaunchKernel(Func, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
                 BlockDim.z, ShMemSize, 0, KernelArgs, nullptr);
#elif defined(ENABLE_HIP)
  DeviceRTErrCheck(hipModuleLaunchKernel(Func, GridDim.x, GridDim.y, GridDim.z,
                                         BlockDim.x, BlockDim.y, BlockDim.z,
                                         ShMemSize, 0, KernelArgs, nullptr));

#endif
  DeviceRTErrCheck(PREFIX(EventRecord(stop)));
  DeviceRTErrCheck(PREFIX(EventSynchronize(stop)));
  DeviceRTErrCheck(PREFIX(EventElapsedTime(&milliseconds, start, stop)));

  DeviceRTErrCheck(PREFIX(DeviceSynchronize()));
  DeviceRTErrCheck(PREFIX(EventDestroy(start)));
  DeviceRTErrCheck(PREFIX(EventDestroy(stop)));
  return milliseconds;
}
