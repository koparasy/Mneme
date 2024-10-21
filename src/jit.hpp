#pragma once

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include "device_types.hpp"

float LaunchFunction(gpu::DeviceModule &Mod, gpu::DeviceFunction &Func,
                     dim3 GridDim, dim3 &BlockDim, uint64_t ShMemSize,
                     void **KernelArgs);

void IRToBackEnd(llvm::Module &M, std::string &CudaArch,
                 llvm::SmallVectorImpl<char> &PTXStr);

void CreateDeviceObject(llvm::StringRef &CodeRepr, gpu::DeviceModule &Mod);

void GetDeviceFunction(gpu::DeviceFunction &Func, gpu::DeviceModule &Mod,
                       llvm::StringRef FunctionName);

void OptimizeIR(llvm::Module &M, unsigned int lvl, std::string &DeviceArch);

void SetLaunchBounds(llvm::Module &M, std::string &FnName, int MaxThreads,
                     int MinBlocks);

llvm::Expected<llvm::orc::ThreadSafeModule> loadIR(llvm::StringRef FileName);

void InitJITEngine();
std::string getArch();
