#pragma once

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace jit {
namespace cuda {
#ifdef ENABLE_CUDA
void LaunchFunction(CUmodule &CUMod, CUfunction &CUFunc, dim3 GridDim,
                    dim3 &BlockDim, uint64_t ShMemSize, void **KernelArgs);

void IRToBackEnd(llvm::Module &M, std::string &CudaArch,
                 llvm::SmallVectorImpl<char> &PTXStr);

void CreateDeviceObject(llvm::StringRef &PTX, CUmodule &CUMod);

void GetDeviceFunction(CUfunction &CUFunc, CUmodule &CUMod,
                       llvm::StringRef FunctionName);

void OptimizeIR(llvm::Module &M, unsigned int lvl, std::string &CudaArch);

void SetLaunchBounds(llvm::Module &M, std::string &FnName, int MaxThreads,
                     int MinBlocks);

llvm::Expected<llvm::orc::ThreadSafeModule> loadIR(llvm::StringRef FileName);

void InitJITEngine();
std::string getArch();

#endif

} // namespace cuda
} // namespace jit
