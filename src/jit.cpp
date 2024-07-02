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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>

#include <iostream>
#include <memory>
#include <string>

#include "jit.hpp"
#include "macro.hpp"

using namespace llvm;

static codegen::RegisterCodeGenFlags CFG;

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

static inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}

static Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, StringRef CPU,
                    CodeGenOpt::Level CGOptLevel = CodeGenOpt::Aggressive) {
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

static void codegenPTX(Module &M, StringRef CudaArch,
                       SmallVectorImpl<char> &PTXStr) {
  auto TMExpected = createTargetMachine(M, CudaArch);
  if (!TMExpected)
    FATAL_ERROR(toString(TMExpected.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  raw_svector_ostream PTXOS(PTXStr);
  TM->addPassesToEmitFile(PM, PTXOS, nullptr, CGFT_AssemblyFile,
                          /* DisableVerify */ false, MMIWP);

  PM.run(M);
}

namespace jit {
namespace cuda {

void InitJITEngine() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

std::string getArch() {
  CUdevice CUDev;
  CUcontext CUCtx;

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
  return CudaArch;
}

Expected<llvm::orc::ThreadSafeModule> loadIR(StringRef FileName) {
  SMDiagnostic Err;
  auto Ctx = std::make_unique<LLVMContext>();
  if (auto M = parseIRFile(FileName, Err, *Ctx)) {
    return llvm::orc::ThreadSafeModule(std::move(M), std::move(Ctx));
  }
  return createSMDiagnosticError(Err);
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

void OptimizeIR(Module &M, unsigned int lvl, std::string &CudaArch) {
  // TODO: Specialize based on lvl
  runOptimizationPassPipeline(M, CudaArch, OptimizationLevel::O3);
}

void IRToBackEnd(Module &M, std::string &CudaArch,
                 SmallVectorImpl<char> &PTXStr) {
  codegenPTX(M, CudaArch, PTXStr);
  // CUDA requires null-terminated PTX.
  PTXStr.push_back('\0');
}

void CreateDeviceObject(StringRef &PTX, CUmodule &CUMod) {
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

void GetDeviceFunction(CUfunction &CUFunc, CUmodule &CUMod,
                       StringRef FunctionName) {
  cuErrCheck(cuModuleGetFunction(&CUFunc, CUMod, FunctionName.data()));
}

void LaunchFunction(CUmodule &CUMod, CUfunction &CUFunc, dim3 GridDim,
                    dim3 &BlockDim, uint64_t ShMemSize, void **KernelArgs) {
  std::cout << "Launching with Grid (" << GridDim.x << "," << GridDim.y << ","
            << GridDim.z << ")\n";
  std::cout << "Launching with Block(" << BlockDim.x << "," << BlockDim.y << ","
            << BlockDim.z << ")\n";

  cuErrCheck(cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                            BlockDim.y, BlockDim.z, ShMemSize, nullptr,
                            KernelArgs, nullptr));
  cudaErrCheck(cudaDeviceSynchronize());
}
} // namespace cuda
} // namespace jit
