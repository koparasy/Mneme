#include "MnemeRecord.hpp"
#include "MnemeRecordHIP.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

using namespace mneme;
using namespace llvm;

MnemeRecorderHIP &MnemeRecorderHIP::instance() {
  static MnemeRecorderHIP Recorder{};
  return Recorder;
}

void MnemeRecorderHIP::extractIR() {
  DBG(Logger::logs("mneme") << "Extracting IR \n");
  constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t Pos = 0;

  for (auto &[Handle, FatbinWrapper] : HandleToBin) {
    const char *Binary = FatbinWrapper->Binary;
    StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      FATAL_ERROR("Error missing magic string");
    Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

    auto Read8ByteIntLE = [](const char *S, size_t Pos) {
      return support::endian::read64le(S + Pos);
    };

    uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
    Pos += 8;
    DBG(Logger::logs("mneme") << "NumberOfbundles " << NumberOfBundles << "\n");

    StringRef DeviceBinary;
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {
      uint64_t Offset = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      uint64_t Size = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      uint64_t TripleSize = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      StringRef Triple(Binary + Pos, TripleSize);
      Pos += TripleSize;

      DBG(Logger::logs("proteus") << "Offset " << Offset << "\n");
      DBG(Logger::logs("proteus") << "Size " << Size << "\n");
      DBG(Logger::logs("proteus") << "TripleSize " << TripleSize << "\n");
      DBG(Logger::logs("proteus") << "Triple " << Triple.str() << "\n");

      if (!Triple.contains("amdgcn"))
        continue;

      DeviceBinary = StringRef(Binary + Offset, Size);
      break;
    }

    Expected<object::ELF64LEFile> DeviceElf =
        object::ELF64LEFile::create(DeviceBinary);
    if (DeviceElf.takeError())
      FATAL_ERROR("Cannot create the device elf");

    auto Sections = DeviceElf->sections();
    if (Sections.takeError())
      FATAL_ERROR("Error reading sections");

    ArrayRef<uint8_t> DeviceBitcode;

    LLVMContext Ctx;

    auto extractModuleFromSection = [&DeviceElf, &Ctx](auto &Section,
                                                       StringRef SectionName) {
      ArrayRef<uint8_t> BitcodeData;
      auto SectionContents = DeviceElf->getSectionContents(Section);
      if (SectionContents.takeError())
        FATAL_ERROR("Error reading section contents");
      BitcodeData = *SectionContents;
      auto Bitcode =
          StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                    BitcodeData.size()};

      SMDiagnostic Err;
      auto M = parseIR(MemoryBufferRef{Bitcode, SectionName}, Err, Ctx);
      if (!M)
        FATAL_ERROR("unexpected");
      return M;
    };

    // We extract bitcode from sections. If there is a .jit.bitcode.lto section
    // due to RDC compilation that's the only bitcode we need, othewise we
    // collect all .jit.bitcode sections.

    SmallVector<std::unique_ptr<Module>> LLVMModules;
    for (auto Section : *Sections) {
      auto SectionName = DeviceElf->getSectionName(Section);
      if (SectionName.takeError())
        FATAL_ERROR("Error reading section name");
      DBG(Logger::logs("proteus")
          << "SectionName " << SectionName.get().str() << "\n");

      if (!SectionName->starts_with(".jit.bitcode"))
        continue;

      auto M = extractModuleFromSection(Section, *SectionName);

      if (SectionName->equals(".jit.bitcode.lto")) {
        LLVMModules.clear();
        LLVMModules.push_back(std::move(M));
        break;
      } else {
        LLVMModules.push_back(std::move(M));
      }
    }

    DenseMap<std::string, Function *> KernelNameToFunction;
    for (auto &Mod : LLVMModules) {
      for (Function &Func : *Mod.get()) {
        // Skip non kernels
        if (Func.getCallingConv() != CallingConv::AMDGPU_KERNEL)
          continue;

        // Can a declarion have a calling conv, if no this is unecessary.
        if (Func.isDeclaration())
          continue;

        KernelNameToFunction[Func.getName().str()] = &Func;
      }
    }

    auto getFuncDescr = [&, this](Function &F) {
      SmallVector<uint64_t, 8> RRInfo;
      auto DL = F.getParent()->getDataLayout();
      for (auto &A : F.args()) {
        // Datatypes such as structs passed by value to kernels are copied
        // into a parameter vector. Over here we test whether an argument is
        // byval, if it is we know on the host side this invocation forwards
        // the arguments by value
        if (A.hasByRefAttr() || A.hasByValAttr()) {
          RRInfo.emplace_back(
              DL.getTypeStoreSize(A.getPointeeInMemoryValueType()));
        } else {
          RRInfo.emplace_back(DL.getTypeStoreSize(A.getType()));
        }
      }
      return RRInfo;
    };

    auto &CurrKernels = HandleToKernels[Handle];
    for (auto &KI : CurrKernels) {
      auto Iter = KernelNameToFunction.find(KI->getName());
      if (Iter == KernelNameToFunction.end())
        FATAL_ERROR("KernelName not in Module");

      Function *KFunc = Iter->second;
      auto FuncArgs = getFuncDescr(*KFunc);
      KI->setArgs(FuncArgs);
    }

    for (auto &Mod : LLVMModules) {
      auto FName = storeModule(*Mod);
      for (auto &KI : CurrKernels) {
        KI->ModuleFiles.push_back(FName);
      }
    }
  }
}

extern "C" {
void __hipRegisterFatBinaryEnd(void *ptr) {
  auto &mneme = MnemeRecorderHIP::instance();
  mneme.registerFatBinEnd(ptr);
}

void **__hipRegisterFatBinary(void *fatbin) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.registerFatBin(static_cast<FatBinaryWrapper_t *>(fatbin));
}

void __hipRegisterVar(void **fatbinHandle, char *hostVar, char *deviceAddress,
                      const char *deviceName, int ext, size_t size,
                      int constant, int global) {
  auto &mneme = MnemeRecorderHIP::instance();
  mneme.registerVar(fatbinHandle, hostVar, deviceAddress, deviceName, ext, size,
                    constant, global);
};

void __hipRegisterFunction(void **fatbinHandle, const char *hostFun,
                           char *deviceFun, const char *deviceName,
                           int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim,
                           dim3 *gDim, int *wSize) {
  auto &mneme = MnemeRecorderHIP::instance();
  mneme.registerFunc(fatbinHandle, hostFun, deviceFun, deviceName, thread_limit,
                     tid, bid, bDim, gDim, wSize);
};

hipError_t hipMalloc(void **ptr, size_t size) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipFree(void *ptr) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipHostFree(void *ptr) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                           void **args, size_t sharedMem, hipStream_t stream) {
  auto &mneme = MnemeRecorderHIP::instance();
  return mneme.rtLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
}
