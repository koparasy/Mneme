//=============================================================================
// FILE:
//    RRPass.cpp
//
// DESCRIPTION:
// Stores LLVM IR in a global variable inside the module and identifies the size
// of arguments of all the device kernels
//
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
// License: MIT
//=============================================================================
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
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
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <iostream>
#include <string>

using namespace llvm;

//-----------------------------------------------------------------------------
// RRPass implementation
//-----------------------------------------------------------------------------
namespace {

struct RRFunctionInfo {
  SmallVector<uint64_t, 8> ArgSize;
};

MapVector<Function *, SmallVector<uint64_t, 8>> RRFunctionInfoMap;

SmallPtrSet<Function *, 16> ModuleDeviceKernels;

static bool isDeviceCompilation(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
    return true;

  return false;
}

static SmallPtrSet<Function *, 16> getDeviceKernels(Module &M) {
  SmallPtrSet<Function *, 16> Kernels;
#if ENABLE_CUDA
  NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

  if (!MD)
    return Kernels;

  for (auto *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
    if (!KindID || KindID->getString() != "kernel")
      continue;

    Function *KernelFn =
        mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    if (!KernelFn)
      continue;

    Kernels.insert(KernelFn);
  }
#elif ENABLE_HIP
  for (Function &F : M)
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL)
      Kernels.insert(&F);
#endif

  return Kernels;
}

static bool isDeviceKernel(const Function *F) {
  if (ModuleDeviceKernels.contains(F))
    return true;

  return false;
}

static void createRRModuleDevice(Module &M) {

  auto ClonedMod = CloneModule(M);
  std::string ModuleIR;
  raw_string_ostream OS(ModuleIR);
  WriteBitcodeToFile(*ClonedMod, OS);
  OS.flush();

  Constant *ClonedModule = ConstantDataArray::get(
      M.getContext(),
      ArrayRef<uint8_t>((const uint8_t *)ModuleIR.data(), ModuleIR.size()));
  auto *GV = new GlobalVariable(
      M, ClonedModule->getType(), /* isConstant */ true,
      GlobalValue::PrivateLinkage, ClonedModule, "record_replay_module");
  appendToUsed(M, {GV});
#if ENABLE_HIP
  // TODO: We need to provide a unique identifier in the sections. Probably
  // prefixing them
  GV->setSection(Twine(".record_replay");
#endif

  return;
}

void visitor(Module &M) {
  if (!isDeviceCompilation(M)) {
    dbgs() << "Not a device compilation exiting... \n";
    return;
  }

  auto DeviceKernels = getDeviceKernels(M);
  auto DL = M.getDataLayout();

  // For every kernel figure out the size of every argument element
  for (auto F : DeviceKernels) {
    SmallVector<uint64_t, 8> RRInfo;
    for (auto &A : F->args()) {
      // Datatypes such as structs passed by value to kernels are copied into a
      // parameter vector. Over here we test whether an argument is byval, if it
      // is we know on the host side this invocation forwards the arguments by
      // value
      if (!A.hasByValAttr()) {
        RRInfo.emplace_back(DL.getTypeStoreSize(A.getType()));
      } else {
        RRInfo.emplace_back(
            DL.getTypeStoreSize(A.getPointeeInMemoryValueType()));
      }
    }
    RRFunctionInfoMap.insert({F, RRInfo});
  }

  for (auto KV : RRFunctionInfoMap) {
    dbgs() << "Function " << KV.first->getName() << "\n";
    for (auto S : KV.second) {
      dbgs() << "Size: " << S << "\n";
    }
  }

  // Append to the global section the size of every argument of the device
  // function. This will be read by the Record/Replay runtime library to infer
  // the sizes of the copies
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  for (auto KV : RRFunctionInfoMap) {
    if (KV.second.size() != 0) {
      ArrayType *RuntimeConstantArrayTy =
          ArrayType::get(Int64Ty, KV.second.size());
      Constant *CA = ConstantDataArray::get(M.getContext(), KV.second);
      auto *GV = new GlobalVariable(M, RuntimeConstantArrayTy, true,
                                    GlobalValue::PrivateLinkage, CA,
                                    "_record_replay_" + KV.first->getName());
      appendToUsed(M, {GV});
    }
  }

  std::string ModuleIR;
  raw_string_ostream OS(ModuleIR);
  WriteBitcodeToFile(M, OS);
  OS.flush();

  // TODO: The IR stored in the module will not contain the global of the module
  // ir. I am wondering whether this will result in alignment issues during
  // replay. We will need to verify
  Constant *IRModule = ConstantDataArray::get(
      M.getContext(),
      ArrayRef<uint8_t>((const uint8_t *)ModuleIR.data(), ModuleIR.size()));
  auto *GV = new GlobalVariable(M, IRModule->getType(), /* isConstant */ true,
                                GlobalValue::PrivateLinkage, IRModule,
                                "__record_replay_ir_module__");
  appendToUsed(M, {GV});
}

// New PM implementation
struct RRPass : PassInfoMixin<RRPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    visitor(M);
    // TODO: is anything preserved?
    return PreservedAnalyses::none();
    // return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation
struct LegacyRRPass : public ModulePass {
  static char ID;
  LegacyRRPass() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    visitor(M);
    // TODO: what is preserved?
    return true;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getRRPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    // We want to register early to allow later for pass order optimization from
    // the extracted IR.
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(RRPass());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "RRPass", LLVM_VERSION_STRING, callback};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize RRPass when added to the pass pipeline on the
// command line, i.e. via '-passes=rr-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getRRPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyRRPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyRRPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-rr-pass'
static RegisterPass<LegacyRRPass>
    X("legacy-rr-pass", "RR Pass",
      false, // Does this pass modify the CFG => false
      false  // Is this a pure analysis pass => false
    );
