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
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <iostream>
#include <string>
#include <utility>

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

StructType *getRecordReplayFuncDescTy(Module &M) {
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  return StructType::create({Int64Ty, PointerType::get(Int64Ty, 0)},
                            "func_info");
}

std::string generateRecordReplayKernelName(StringRef Name) {
  return "_record_replay_func_info_" + Name.str();
}

/* This function iterates over all functions in the module and detects which are
 * "kernels" (a.k.a callable from host). For every kernel it creates a global
 * variable that store the number of arguments of the kernel and the size (in
 * bytes) of every argument
 *
 * Further, it creates a global variable that contains the LLVM IR of the module
 * and the the size of the IR. On the host side we will read in these values to
 * load the LLVM IR.
 */
void deviceInstrumentation(Module &M) {
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
  StructType *FunctionInfoTy = getRecordReplayFuncDescTy(M);
  for (auto KV : RRFunctionInfoMap) {
    if (KV.second.size() != 0) {
      ArrayType *RuntimeConstantArrayTy =
          ArrayType::get(Int64Ty, KV.second.size());
      Constant *CA = ConstantDataArray::get(M.getContext(), KV.second);
      auto *Elements = new GlobalVariable(
          M, RuntimeConstantArrayTy, true, GlobalValue::PrivateLinkage, CA,
          "_record_replay_elements_" + KV.first->getName());

      Constant *Index = ConstantInt::get(Int64Ty, 0, false);
      Constant *GVPtr =
          ConstantExpr::getGetElementPtr(Int64Ty, Elements, Index);
      Constant *NumElems = ConstantInt::get(Int64Ty, KV.second.size(), false);
      Constant *CS = ConstantStruct::get(FunctionInfoTy, {NumElems, GVPtr});
      auto *GV = new GlobalVariable(
          M, FunctionInfoTy, true, GlobalValue::PrivateLinkage, CS,
          generateRecordReplayKernelName(KV.first->getName()));
      appendToUsed(M, {GV});
    } else {
      Constant *NumElems = ConstantInt::get(Int64Ty, KV.second.size(), false);
      Constant *CS = ConstantStruct::get(
          FunctionInfoTy,
          {NumElems, ConstantPointerNull::get(PointerType::get(Int64Ty, 0))});
      auto *GV = new GlobalVariable(
          M, FunctionInfoTy, true, GlobalValue::PrivateLinkage, CS,
          generateRecordReplayKernelName(KV.first->getName()));
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
                                "_record_replay_ir_module_");

  Type *Int8Ty = Type::getInt64Ty(M.getContext());
  StructType *ModuleIRTy =
      StructType::create({Int64Ty, PointerType::get(Int8Ty, 0)}, "module_info");

  Constant *IRSize = ConstantInt::get(Int64Ty, ModuleIR.size(), false);

  Constant *Index = ConstantInt::get(Int64Ty, 0, false);
  Constant *IRPtr = ConstantExpr::getGetElementPtr(Int8Ty, GV, Index);

  Constant *CS = ConstantStruct::get(ModuleIRTy, {IRSize, IRPtr});

  // The '.' character results in weird ptxas behavior. We replace it.
  std::string IRName = "_record_replay_descr_" + M.getSourceFileName();
  std::replace(IRName.begin(), IRName.end(), '.', '_');
  std::replace(IRName.begin(), IRName.end(), '/', '_');

  auto *GVIR = new GlobalVariable(M, ModuleIRTy, true,
                                  GlobalValue::PrivateLinkage, CS, IRName);

  appendToUsed(M, {GVIR});

  errs() << "DEVICE MODULE: \n";
  errs() << M << "\n";
}

static bool HasDeviceKernelCalls(Module &M) {
#if ENABLE_HIP
  Function *LaunchKernelFn = M.getFunction("hipLaunchKernel");
#elif ENABLE_CUDA
  Function *LaunchKernelFn = M.getFunction("cudaLaunchKernel");
#endif
  if (!LaunchKernelFn)
    return false;

  return true;
}

StringRef getGlobalString(GlobalVariable *GV) {
  auto cInitializer = dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (cInitializer)
    return cInitializer->getAsCString();
  return StringRef("");
}

static std::pair<GlobalVariable *, GlobalVariable *>
CreateGlobalVariable(Module &M, StringRef globalName) {
  // Generate the global stub variable.
  StructType *FunctionInfoTy = getRecordReplayFuncDescTy(M);

  auto ASpace = M.getDataLayout().getDefaultGlobalsAddressSpace();
  auto *GVStub = new GlobalVariable(
      M, FunctionInfoTy, false /*isConstant=*/, GlobalValue::InternalLinkage,
      UndefValue::get(FunctionInfoTy), globalName, nullptr,
      llvm::GlobalValue::NotThreadLocal, ASpace);

  GVStub->setAlignment(MaybeAlign(8));

  // Generate the global variable storing the name of the global stub variable
  Constant *VName = ConstantDataArray::get(
      M.getContext(), ArrayRef<uint8_t>((const uint8_t *)globalName.data(),
                                        globalName.size() + 1));
  auto *GVName = new GlobalVariable(M, VName->getType(), /* isConstant */ true,
                                    GlobalValue::InternalLinkage, VName);
  GVName->setAlignment(MaybeAlign(1));

  return std::make_pair(GVStub, GVName);
}

static void RegisterGlobalVariable(Module &M, Value *fatBinHandle,
                                   Instruction *IP, StringRef globalName) {
#if ENABLE_HIP
#error pending implementation
#elif ENABLE_CUDA
  Function *RegisterVariable = M.getFunction("__cudaRegisterVar");
#endif
  auto dSize = M.getDataLayout().getTypeStoreSize(getRecordReplayFuncDescTy(M));
  auto globals = CreateGlobalVariable(M, globalName);
  IRBuilder<> Builder(IP);
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Constant *Ext = ConstantInt::get(Int32Ty, 0, false);
  Constant *Size = ConstantInt::get(Int64Ty, dSize, false);
  Constant *Const = ConstantInt::get(Int32Ty, 0, false);
  Constant *Global = ConstantInt::get(Int32Ty, 0, false);
  FunctionCallee fnType(RegisterVariable);
  Builder.CreateCall(fnType, {fatBinHandle, globals.first, globals.second,
                              globals.second, Ext, Size, Const, Global});
}

static DenseMap<StringRef, std::pair<GlobalVariable *, GlobalVariable *>>
InstantiateGlobalVariables(Module &M) {
#if ENABLE_HIP
  Function *RegisterFunction = M.getFunction("__hipRegisterFunction");
  assert(false && "Pending implementation");
#elif ENABLE_CUDA
  Function *RegisterFunction = M.getFunction("__cudaRegisterFunction");
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

  DenseMap<StringRef, std::pair<GlobalVariable *, GlobalVariable *>>
      KernelNameMap;
  if (RegisterFunction) {
    constexpr int KernelOperand = 2;
    for (User *Usr : RegisterFunction->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        GlobalVariable *GV =
            dyn_cast<GlobalVariable>(CB->getArgOperand(KernelOperand));
        assert(GV && "Expected global variable as kernel name operand");
        auto cName = getGlobalString(GV);
        assert(!cName.empty() && "Expected valid kernel stub key");
        RegisterGlobalVariable(M, CB->getOperand(0),
                               CB->getNextNonDebugInstruction(),
                               generateRecordReplayKernelName(cName));
      }
  }

  return KernelNameMap;
}

size_t getFatBinSize(Module &M, GlobalVariable *FatbinWrapper) {
#if ENABLE_CUDA
  ConstantStruct *C = dyn_cast<ConstantStruct>(FatbinWrapper->getInitializer());
  assert(C->getType()->getNumElements() &&
         "Expected four fields in fatbin wrapper struct");
  constexpr int FatbinField = 2;
  auto *Fatbin = C->getAggregateElement(FatbinField);
  GlobalVariable *FatbinGV = dyn_cast<GlobalVariable>(Fatbin);
  assert(FatbinGV && "Expected global variable for the fatbin object");
  ArrayType *ArrayTy =
      dyn_cast<ArrayType>(FatbinGV->getInitializer()->getType());
  assert(ArrayTy && "Expected array type of the fatbin object");
  assert(ArrayTy->getElementType() == Type::getInt8Ty(M.getContext()) &&
         "Expected byte type for array type of the fatbin object");
  size_t FatbinSize = ArrayTy->getNumElements();
  return FatbinSize;
#elif ENABLE_HIP
  return 0;
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif
}

GlobalVariable *getFatBinWrapper(Module &M) {
#if ENABLE_HIP
  GlobalVariable *FatbinWrapper =
      M.getGlobalVariable("__hip_fatbin_wrapper", true);
#elif ENABLE_CUDA
  GlobalVariable *FatbinWrapper =
      M.getGlobalVariable("__cuda_fatbin_wrapper", true);
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif
  return FatbinWrapper;
}

FunctionType *getRRRuntimeCallFnTy(Module &M) {
  Type *voidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  FunctionType *RREntryFn =
      FunctionType::get(voidTy, {VoidPtrTy, Int64Ty}, /*isVarArg=*/false);
  return RREntryFn;
}

void RegisterFatBinary(Module &M) {
#if ENABLE_HIP
  assert(false && "Pending implementation");
#elif ENABLE_CUDA
  Function *RegisterFatBinaryFn = M.getFunction("__cudaRegisterFatBinary");
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

  if (RegisterFatBinaryFn) {
    for (User *Usr : RegisterFatBinaryFn->users()) {
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        errs() << "CallBack\n";
        errs() << *CB << "\n";
        IRBuilder<> Builder(CB->getNextNonDebugInstruction());
        auto *FatBin = getFatBinWrapper(M);
        size_t FatBinSize = getFatBinSize(M, FatBin);
        FunctionType *RRFnTy = getRRRuntimeCallFnTy(M);
        FunctionCallee RRFn =
            M.getOrInsertFunction("__rr_register_fat_binary", RRFnTy);
        Builder.CreateCall(RRFn, {FatBin, Builder.getInt64(FatBinSize)});
        // Once we insert a register we can break. There is a single
        // registration per translation unit.
        // TODO: Verify this in the case of multiple device files
        break;
      }
    }
  }
}

void RegisterLLVMIRVariable(Module &M) {
#if ENABLE_HIP
  assert(false && "Pending implementation");
#elif ENABLE_CUDA
  Function *RegisterFatBinaryFn = M.getFunction("__cudaRegisterFatBinary");
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

  if (RegisterFatBinaryFn) {
    for (User *Usr : RegisterFatBinaryFn->users()) {
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        std::string IRName = "_record_replay_descr_" + M.getSourceFileName();
        // I need to do this, cause ptxas does not like '.' and we cannot find
        // the respective symbols
        std::replace(IRName.begin(), IRName.end(), '.',
                     '_'); // replace all 'x' to 'y'
                           //
        std::replace(IRName.begin(), IRName.end(), '/', '_');
        RegisterGlobalVariable(M, CB, CB->getNextNonDebugInstruction(), IRName);

        break;
      }
    }
  }
}

void hostInstrumentation(Module &M) {
  if (!HasDeviceKernelCalls(M)) {
    errs() << "Module " << M.getSourceFileName()
           << " Does not have kernel functions\n";
    return;
  }
  // We register all device globals so that we can access them.
  auto KernelNameMap = InstantiateGlobalVariables(M);

  RegisterLLVMIRVariable(M);
  RegisterFatBinary(M);
  errs() << M << "\n";

  //
}

void visitor(Module &M) {
  if (!isDeviceCompilation(M)) {
    hostInstrumentation(M);
    return;
  }
  deviceInstrumentation(M);
  return;
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
