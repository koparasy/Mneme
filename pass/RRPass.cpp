//=============================================================================
// Part of the Mneme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libRRPass.dylib -legacy-rr-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libRRPass.dylib -passes="rr-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ELF.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileSystem/UniqueID.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include "../common/Logger.hpp"
#include <iostream>
#include <string>

#define DEBUG_TYPE "jitpass"
#ifdef ENABLE_DEBUG
#define DEBUG(x) x
constexpr auto debug_build = true;
#else
constexpr auto debug_build = false;
#define DEBUG(x)
#endif

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

#if ENABLE_HIP
constexpr char const *RegisterFunctionName = "__hipRegisterFunction";
constexpr char const *LaunchFunctionName = "hipLaunchKernel";
constexpr char const *RegisterVarName = "__hipRegisterVar";
constexpr char const *RegisterFatBinaryName = "__hipRegisterFatBinary";
#elif ENABLE_CUDA
constexpr char const *RegisterFunctionName = "__cudaRegisterFunction";
constexpr char const *LaunchFunctionName = "cudaLaunchKernel";
constexpr char const *RegisterVarName = "__cudaRegisterVar";
constexpr char const *RegisterFatBinaryName = "__cudaRegisterFatBinary";
#else
constexpr char const *RegisterFunctionName = nullptr;
constexpr char const *LaunchFunctionName = nullptr;
constexpr char const *RegisterVarName = nullptr;
constexpr char const *RegisterFatBinaryName = nullptr;
#endif

using namespace llvm;
using namespace mneme;

//-----------------------------------------------------------------------------
// MnemeRegisterIRPass implementation
//-----------------------------------------------------------------------------
namespace {

void dump(Module &M, StringRef device, StringRef phase) {

  if (!debug_build)
    return;

  std::filesystem::path ModulePath(M.getSourceFileName());
  std::filesystem::path filename(M.getSourceFileName());
  std::string rrBC(
      Twine(filename.filename().string() + "." + device + "." + phase + ".bc")
          .str());
  std::error_code EC;
  raw_fd_ostream OutBC(rrBC, EC);
  if (EC)
    throw std::runtime_error("Cannot open device code " + rrBC);
  OutBC << M;
  OutBC.close();
}

class MnemePassImpl {
public:
  MnemePassImpl(Module &M) {}

  bool isDeviceCompilation(Module &M) {
    Triple TargetTriple(M.getTargetTriple());
    DEBUG(Logger::logs("mneme-pass")
          << "TargetTriple " << M.getTargetTriple() << "\n");
    if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
      return true;

    return false;
  }

  bool run(Module &M, bool IsLTO) {
    // ==================
    // Device compilation
    // ==================
    if (isDeviceCompilation(M)) {
      dump(M, "device", IsLTO ? "lto-before-mneme" : "before-mneme");
      emitJitModuleDevice(M, IsLTO);
      dump(M, "device", IsLTO ? "lto-before-mneme" : "after-mneme");
      return true;
    }

    // ================
    // Host compilation
    // ================
    dump(M, "host", IsLTO ? "lto-before-proteus" : "before-proteus");

    if (verifyModule(M, &errs()))
      FATAL_ERROR("Broken original module found, compilation aborted!");

    dump(M, "host", IsLTO ? "lto-before-mneme" : "after-mneme");

    return true;
  }

private:
  std::string getJitBitcodeUniqueName(Module &M) {
    llvm::sys::fs::UniqueID ID;
    if (auto EC = llvm::sys::fs::getUniqueID(M.getSourceFileName(), ID))
      FATAL_ERROR("Could not get unique id");

    SmallString<64> Out;
    llvm::raw_svector_ostream OutStr(Out);
    OutStr << "_mneme_bitcode" << llvm::format("_%x", ID.getDevice())
           << llvm::format("_%x", ID.getFile());

    return std::string(Out);
  }

  void emitJitModuleDevice(Module &M, bool IsLTO) {
    std::string BitcodeStr;
    raw_string_ostream OS(BitcodeStr);
    WriteBitcodeToFile(M, OS);

    std::string GVName =
        (IsLTO ? "__jit_bitcode_lto" : getJitBitcodeUniqueName(M));
    //  NOTE: HIP compilation supports custom section in the binary to store
    //  the IR. CUDA does not, hence we parse the IR by reading the global
    //  from the device memory.
    Constant *DeviceModule = ConstantDataArray::get(
        M.getContext(), ArrayRef<uint8_t>((const uint8_t *)BitcodeStr.data(),
                                          BitcodeStr.size()));
    auto *GV =
        new GlobalVariable(M, DeviceModule->getType(), /* isConstant */ true,
                           GlobalValue::ExternalLinkage, DeviceModule, GVName);
    appendToUsed(M, {GV});
    GV->setSection(".jit.bitcode" + (IsLTO ? ".lto" : getUniqueModuleId(&M)));
    DEBUG(Logger::logs("proteus-pass")
          << "Emit jit bitcode GV " << GVName << "\n");
  }
};

// New PM implementation.
struct MnemePass : PassInfoMixin<MnemePass> {
  MnemePass(bool IsLTO) : IsLTO(IsLTO) {}
  bool IsLTO;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    MnemePassImpl Mneme{M};

    bool Changed = Mneme.run(M, IsLTO);
    if (Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for
  // functions decorated with the optnone LLVM attribute. Note that clang -O0
  // decorates all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation.
struct LegacyMnemePass : public ModulePass {
  static char ID;
  LegacyMnemePass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    MnemePassImpl Mneme{M};
    bool Changed = Mneme.run(M, false);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getMnemePassPluginInfo() {
  const auto Callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime. We choose after early
    // simplifications which should avoid inlining and present a reasonably
    // analyzable IR module.

    // NOTE: For device jitting it should be possible to register the pass late
    // to reduce compilation time and does lose the kernel due to inlining.
    // However, there are linking errors, working assumption is that the hiprtc
    // linker cannot re-link already linked device libraries and aborts.

    // PB.registerPipelineStartEPCallback(
    // PB.registerOptimizerLastEPCallback(
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(MnemePass{false});
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(MnemePass{true});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "MnemePass", LLVM_VERSION_STRING, Callback};
}

// TODO: use by proteus-jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize ProteusJitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=proteus-jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMnemePassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyMnemePass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyProteusJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyMnemePass>
    X("legacy-mneme-pass", "Mneme Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
