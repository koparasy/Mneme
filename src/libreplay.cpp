#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#define DEVICE_FUNC(x) "cuda" x
#define PREFIX_UU(x) __cuda##x
#define PREFIX(x) cuda##x

#elif defined(ENABLE_HIP)

#include <hip/hip_runtime.h>

#define DEVICE_FUNC(x) "hip" x
#define PREFIX_UU(x) __hip##x
#define PREFIX(x) hip##x

#else
#error "Must define ENABLE_CUDA or ENABLE_HIP when building"
#endif

#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#if ENABLE_REPLAY_OPT
#include "replay_opt.hpp"
#endif

// Overloaded functions
static PREFIX(Error_t) (*deviceLaunchKernelInternal)(
    const void *func, dim3 gridDim, dim3 blockDim, void **args,
    size_t sharedMem, PREFIX(Stream_t) stream) = nullptr;
static PREFIX(Error_t) (*deviceMallocInternal)(void **ptr, size_t size);
static PREFIX(Error_t) (*deviceMallocHostInternal)(void **ptr, size_t size);
static PREFIX(Error_t) (*deviceMallocManagedInternal)(void **ptr, size_t size,
                                                      unsigned int flags);
static PREFIX(Error_t) (*deviceFreeInternal)(void *devPtr);
static void (*__deviceRegisterVarInternal)(void **fatDevbinHandle,
                                           char *hostVar, char *deviceAddress,
                                           const char *deviceName, int ext,
                                           size_t size, int constant,
                                           int global);
static void **(*__deviceRegisterFatBinaryInternal)(void *fatDevbin);
static void (*__deviceRegisterFunctionInternal)(
    void **fatDevbinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);

// Reverse eng. fat binary format, unused for now.
struct CudaRegisterFatBinaryArguments {
  int magic_number;
  int version;
  const void *binary;
  void *unused;
};

struct MeasureInfo {
  dim3 GridDim;
  dim3 BlockDim;
  double Time;
};

// Singleton class that initializes wrapping.
class Wrapper {
public:
  static Wrapper *instance() {
    static Wrapper w;
    return &w;
  };

  //~Wrapper() {
  //  for (auto &Entry : Tunings) {
  //    const std::pair<const void *, int> Key = Entry.first;
  //    std::pair<dim3, dim3> Values = Entry.second;

  //    DEBUG(std::cout << "Func " << Key.first << " Dim " << Key.second
  //                    << " Best size Grid " << Values.first.x << ","
  //                    << Values.first.y << "," << Values.first.y << ","
  //                    << Values.first.z << " Block " << Values.second.x << ","
  //                    << Values.second.y << "," << Values.second.z << "\n");
  //  }
  //}

  void getDeviceInfo(int &WarpSize, int &MultiProcessorCount,
                     int &MaxGridSizeX) {
    WarpSize = this->WarpSize;
    MultiProcessorCount = this->MultiProcessorCount;
    MaxGridSizeX = this->MaxGridSizeX;

    return;
  }

  void saveDeviceMemory() {
    Copies.resize(DeviceMemoryMap.size());
    Kinds.resize(DeviceMemoryMap.size());

    size_t idx = 0;
    for (auto &Entry : DeviceMemoryMap) {
      void *ptr = Entry.first;
      size_t size = Entry.second;

      // Use device for saving memory.
      if (deviceMallocInternal(&Copies[idx], size) == PREFIX(Success)) {
        // deviceMemcpyAsync(Copies[idx], ptr, size,
        // PREFIX(MemcpyDeviceToDevice), stream);
        PREFIX(Memcpy)
        (Copies[idx], ptr, size, PREFIX(MemcpyDeviceToDevice));
        DEBUG(std::cout << "D2D copy ptr " << ptr << " size " << size << " to "
                        << Copies[idx] << "\n");
        Kinds[idx] = PREFIX(MemcpyDeviceToDevice);
      } else {
        // Fallback to host.
        Copies[idx] = malloc(size);
        assert(Copies[idx] && "Expected non-null copy ptr");
        DEBUG(std::cout << "Fallback host copy ptr " << ptr << " size " << size
                        << " to " << Copies[idx] << "\n");
        // deviceMemcpyAsync(Copies[idx], ptr, size,
        // PREFIX(MemcpyDeviceToHost), stream);
        PREFIX(Memcpy)(Copies[idx], ptr, size, PREFIX(MemcpyDeviceToHost));
        // Note this should be the action when restoring memory to device
        // from the host.
        Kinds[idx] = PREFIX(MemcpyHostToDevice);
      }

      idx++;
    }

    // NOTE: Synchronization is needed to avoid perturbation to kernel
    // execution time measurements.
    PREFIX(DeviceSynchronize)();
  }

  void restoreDeviceMemory() {
    size_t idx = 0;
    for (auto &Entry : DeviceMemoryMap) {
      void *ptr = Entry.first;
      size_t size = Entry.second;

      // deviceMemcpyAsync(ptr, Copies[idx], size, Kinds[idx], stream);
      PREFIX(Memcpy)(ptr, Copies[idx], size, Kinds[idx]);
      idx++;
    }

    // NOTE: Synchronization is needed to avoid perturbation to kernel
    // execution time measurements.
    PREFIX(DeviceSynchronize)();
  }

  bool matchWhitelist(const void *Func) {
    if (SymbolWhiteList.empty())
      return true;

    for (std::string &Symbol : SymbolWhiteList)
      if (SymbolTable[Func] == Symbol)
        return true;

    return false;
  }

  bool matchExclList(const void *Func) {
    if (SymbolExclList.empty())
      return false;

    for (std::string &Symbol : SymbolExclList)
      if (SymbolTable[Func].find(Symbol) != std::string::npos) {
        DEBUG(std::cout << "Func " << SymbolTable[Func] << " contains "
                        << Symbol << "\n");
        return true;
      }

    return false;
  }

#if ENABLE_FAST_MODE
  void updateTunings(const void *Func, int Dim, dim3 GridDim, dim3 BlockDim,
                     double Time) {
    if (!Tunings.count({Func, Dim})) {
      Tunings[{Func, Dim}] = {GridDim, BlockDim, Time};
      return;
    }

    if (Tunings[{Func, Dim}].Time > Time)
      Tunings[{Func, Dim}] = {GridDim, BlockDim, Time};
  }
#endif

  bool isTunable(const void *Func, int &MaxThreadsPerBlock, int Dim) {
    PREFIX(FuncAttributes) Attr;
    PREFIX(FuncGetAttributes)(&Attr, Func);
    DEBUG(printf("deviceFuncGetAttributes sharedSizeBytes %zu numRegs %d "
                 "maxThreadsPerBlock %d\n",
                 Attr.sharedSizeBytes, Attr.numRegs, Attr.maxThreadsPerBlock);)
    MaxThreadsPerBlock = Attr.maxThreadsPerBlock;

#if ENABLE_FAST_MODE
    if (Tunings.count({Func, Dim}))
      return false;
#endif

    // TODO: Assumes that using shared memory enforces a hard limit on
    // dimensions.
    bool IsSharedZero = (Attr.sharedSizeBytes == 0);
    bool IsTunable =
        (IsSharedZero and matchWhitelist(Func) and not matchExclList(Func));
    DEBUG(std::cout << "IsTunable " << IsTunable << "\n");

    return IsTunable;
  }

  void freeCopies() {
    for (size_t idx = 0; idx < Copies.size(); ++idx) {
      if (Kinds[idx] == PREFIX(MemcpyDeviceToDevice)) {
        DEBUG(std::cout << "deviceFree ptr " << Copies[idx] << "\n");
        deviceFreeInternal(Copies[idx]);
      } else {
        DEBUG(std::cout << "host free ptr " << Copies[idx] << "\n");
        free(Copies[idx]);
      }
    }
  }

  // GlobalsMap stores a mapping of host addr to size, will convert to device
  // pointers using <cuda|hip>GetSymbolAddress.
  std::unordered_map<void *, size_t> GlobalsMap;
  // DeviceMemoryMap holds all the pointers to device memory that need to be
  // copied/restored.
  std::unordered_map<void *, size_t> DeviceMemoryMap;
  std::unordered_map<const void *, std::string> SymbolTable;
#if ENABLE_FAST_MODE
  // Maps <func, Dim> to fastest found <gridDim, blockDim, time>.
  std::map<std::pair<const void *, int>, struct MeasureInfo> Tunings;
#endif
  std::vector<void *> Copies;
  std::vector<PREFIX(MemcpyKind)> Kinds;
  std::vector<std::string> SymbolWhiteList;
  std::vector<std::string> SymbolExclList;

private:
  void *device_runtime_handle;
  int WarpSize;
  int MultiProcessorCount;
  int MaxGridSizeX;

  Wrapper() {
#ifdef ENABLE_CUDA
    device_runtime_handle = dlopen("libcudart.so", RTLD_NOW);
#else
    device_runtime_handle = dlopen("libamdhip64.so", RTLD_NOW);
#endif
    assert(device_runtime_handle && "Expected non-null");

    // Redirect overloaded device runtime functions.
    reinterpret_cast<void *&>(deviceLaunchKernelInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("LaunchKernel"));
    assert(deviceLaunchKernelInternal && "Expected non-null");
#if !defined(ENABLE_REPLAY_OPT)
    reinterpret_cast<void *&>(deviceMallocInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("Malloc"));
    assert(deviceMallocInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceMallocHostInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("MallocHost"));
    assert(deviceMallocHostInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceMallocManagedInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("MallocManaged"));
    assert(deviceMallocManagedInternal && "Expected non-null");
    reinterpret_cast<void *&>(deviceFreeInternal) =
        dlsym(device_runtime_handle, DEVICE_FUNC("Free"));
    assert(deviceFreeInternal && "Expected non-null");

    reinterpret_cast<void *&>(__deviceRegisterVarInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterVar"));
    assert(__deviceRegisterVarInternal && "Expected non-null");
    reinterpret_cast<void *&>(__deviceRegisterFatBinaryInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFatBinary"));
    assert(__deviceRegisterFatBinaryInternal && "Expected non-null");
    reinterpret_cast<void *&>(__deviceRegisterFunctionInternal) =
        dlsym(device_runtime_handle, "__" DEVICE_FUNC("RegisterFunction"));
    assert(__deviceRegisterFunctionInternal && "Expected non-null");
    DEBUG(printf("=== Library inited\n");)

    // Gather device info.
    int DeviceId;
    PREFIX(GetDevice(&DeviceId));
    PREFIX(DeviceGetAttribute)
    (&WarpSize,
    // NOTE: very ugly, thank you vendors!
#ifdef ENABLE_CUDA
     cudaDevAttrWarpSize,
#else
     hipDeviceAttributeWarpSize,
#endif
     DeviceId);
    PREFIX(DeviceGetAttribute)
    (&MultiProcessorCount,
#ifdef ENABLE_CUDA
     cudaDevAttrMultiProcessorCount,
#else
     hipDeviceAttributeMultiprocessorCount,
#endif
     DeviceId);
    // TODO: We use the x-dimension for now, consider adding y,z if needed.
    PREFIX(DeviceGetAttribute)
    (&MaxGridSizeX,
#ifdef ENABLE_CUDA
     cudaDevAttrMaxGridDimX,
#else
     hipDeviceAttributeMaxGridDimX,
#endif
     DeviceId);

    // Gather the symbols whitelist.
    const char *EnvVarSymbols = std::getenv("LIBREPLAY_SYMBOLS");
    std::string Symbols = (EnvVarSymbols ? EnvVarSymbols : "");
    DEBUG(std::cout << "Symbols EnvVar " << Symbols << "\n");

    if (!Symbols.empty())
      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
        endpos = Symbols.find(',', pos);
        SymbolWhiteList.push_back(Symbols.substr(pos, endpos));
        DEBUG(std::cout << "Symbol: " << Symbols.substr(pos, endpos) << "\n");
        pos = endpos + 1;
      }

    // Gather the symbols exclusion list.
    EnvVarSymbols = std::getenv("LIBREPLAY_EXCL_SYMBOLS");
    Symbols = (EnvVarSymbols ? EnvVarSymbols : "");
    DEBUG(std::cout << "Symbols Excl. EnvVar " << Symbols << "\n");

    if (!Symbols.empty())
      for (size_t pos = 0, endpos = 0; endpos != std::string::npos;) {
        endpos = Symbols.find(',', pos);
        SymbolExclList.push_back(Symbols.substr(pos, endpos));
        DEBUG(std::cout << "Symbol excl.: " << Symbols.substr(pos, endpos)
                        << "\n");
        pos = endpos + 1;
      }
#endif
  }
};

class CsvWriter {
public:
  static CsvWriter *instance() {
    static CsvWriter CSVW;
    return &CSVW;
  }
  void PrintCSVRow(std::size_t KernelCounter, const void *Addr,
                   const char *Fname, dim3 GridDim, dim3 BlockDim, int Dim,
                   int MaxGrid, bool IsOriginal, double Time) {
    // TODO: find less ugly way to avoid printing the function name in
    // output.
#ifdef OUTPUT_FUNCTION_NAME
    fprintf(fp, "%lu,%p,%s,%u,%u,%u,%u,%u,%u,%u,%d,%s,%3.3f\n", KernelCounter,
            Addr, Fname, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
            BlockDim.y, BlockDim.z, Dim, MaxGrid,
            (IsOriginal ? "true" : "false"), Time);
#else
    fprintf(fp, "%lu,%p,%u,%u,%u,%u,%u,%u,%u,%d,%s,%3.3f\n", KernelCounter,
            Addr, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
            BlockDim.z, Dim, MaxGrid, (IsOriginal ? "true" : "false"), Time);
#endif
  }

private:
  FILE *fp = nullptr;
  CsvWriter() {
    fp = fopen("libreplay-output.csv", "w");
    if (!fp)
      throw std::runtime_error("Error creating output csv file");
#ifdef OUTPUT_FUNCTION_NAME
    fprintf(fp, "kernel_id,addr,func,gridx,gridy,gridz,blockx,blocky,"
                "blockz,dim,maxgrid,original,time\n");
#else
    fprintf(fp, "kernel_id,addr,gridx,gridy,gridz,blockx,blocky,blockz,dim,"
                "maxgrid,original,time\n");
#endif
  };

  ~CsvWriter() { fclose(fp); }
};

class OptWriter {
public:
  static OptWriter *instance() {
    static OptWriter CSVW;
    return &CSVW;
  }
  void WriteOpt(dim3 GridDim, dim3 BlockDim) {
    fprintf(fp, "{ {%d, %d, %d}, {%d, %d, %d} },\n", GridDim.x, GridDim.y,
            GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z);
  }

private:
  FILE *fp = nullptr;
  OptWriter() {
    fp = fopen("optimal.inc", "w");
    if (!fp)
      throw std::runtime_error("Error creating output optimal inc file");
  };

  ~OptWriter() { fclose(fp); }
};

// Overload implementations.
extern "C" {

#if !defined(ENABLE_REPLAY_OPT)
void **PREFIX_UU(RegisterFatBinary)(void *fatCubin) {
  Wrapper *W = Wrapper::instance();
  void **ret = __deviceRegisterFatBinaryInternal(fatCubin);
  // auto *args = static_cast<CudaRegisterFatBinaryArguments *>(fatCubin);

  for (auto &Entry : W->GlobalsMap) {
    void *symbol = Entry.first;
    size_t size = Entry.second;

    void *devPtr;
    auto ret = PREFIX(GetSymbolAddress)(&devPtr, symbol);
    assert(ret == PREFIX(Success) && "Error");
    assert(W->DeviceMemoryMap.count(devPtr) == 0 &&
           "Expected non-duplicate entries in device memory");
    W->DeviceMemoryMap.insert({devPtr, size});
  }
  return ret;
}

void PREFIX_UU(RegisterVar)(void **fatCubinHandle, char *hostVar,
                            char *deviceAddress, const char *deviceName,
                            int ext, size_t size, int constant, int global) {
  Wrapper *W = Wrapper::instance();
  __deviceRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress,
                              deviceName, ext, size, constant, global);
  DEBUG(std::cout << "hostVar " << (void *)hostVar << " deviceAddr "
                  << (void *)deviceAddress << " deviceName " << deviceName
                  << " ext " << ext << " size " << size << " constant "
                  << constant << " global " << global << "\n";)

  if (constant)
    return;

  assert(W->GlobalsMap.count(hostVar) == 0 &&
         "Expected non-duplicate Entry in globals");
  W->GlobalsMap[hostVar] = size;
}

void PREFIX_UU(RegisterFunction)(void **fatCubinHandle, const char *hostFun,
                                 char *deviceFun, const char *deviceName,
                                 int thread_limit, uint3 *tid, uint3 *bid,
                                 dim3 *bDim, dim3 *gDim, int *wSize) {
  Wrapper *W = Wrapper::instance();
  DEBUG(std::cout << "Register func hostFun " << (void *)hostFun
                  << " deviceFun " << deviceFun << "\n";)
  int status;
#ifdef ENABLE_CXX_DEMANGLE
  char *demangled = abi::__cxa_demangle(deviceFun, nullptr, nullptr, &status);
  std::string func_name = demangled;
  free(demangled);

  size_t begin = func_name.find(" ") + 1;
  DEBUG(std::cout << "func_name " << func_name << "\n");
  DEBUG(std::cout << "find < " << func_name.find('<') << "\n");
  DEBUG(std::cout << "find ( " << func_name.find('(') << "\n");
  size_t end = std::min(func_name.find('<'), func_name.find('('));
  // Demangle sometimes includes the return type, others doesn't.
  // Reset to 0 if goes beyond end.
  if (begin > end)
    begin = 0;
  W->SymbolTable[(void *)hostFun] = func_name.substr(begin, end - begin);
  DEBUG(std::cout << "Found func " << W->SymbolTable[(void *)hostFun] << "\n");
  // W->SymbolTable[(void *)hostFun] = demangled;
#else
  W->SymbolTable[(void *)hostFun] = deviceFun;
#endif
  __deviceRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun,
                                   deviceName, thread_limit, tid, bid, bDim,
                                   gDim, wSize);
}

PREFIX(Error_t) PREFIX(Malloc)(void **ptr, size_t size) {
  static int called = 0;
  Wrapper *W = Wrapper::instance();
  auto ret = deviceMallocInternal(ptr, size);
  assert(W->DeviceMemoryMap.count(ptr) == 0 &&
         "Expected non-duplicate entries in device memory");
  W->DeviceMemoryMap.insert({*ptr, size});
  DEBUG(for (auto &Entry
             : W->DeviceMemoryMap) std::cout
        << "Malloc Entry ptr " << Entry.first << " size " << Entry.second
        << "\n");
  DEBUG(std::cout << "Called " << called << "\n");
  called++;

  return ret;
}

PREFIX(Error_t) PREFIX(MallocHost)(void **ptr, size_t size) {
  Wrapper *W = Wrapper::instance();
  auto ret = deviceMallocHostInternal(ptr, size);
  assert(W->DeviceMemoryMap.count(ptr) == 0 &&
         "Expected non-duplicate entries in device memory");
  W->DeviceMemoryMap.insert({*ptr, size});
  DEBUG(for (auto &Entry
             : W->DeviceMemoryMap) std::cout
        << "MallocHost Entry ptr " << Entry.first << " size " << Entry.second
        << "\n");

  return ret;
}

PREFIX(Error_t)
PREFIX(MallocManaged)(void **ptr, size_t size, unsigned int flags) {
  Wrapper *W = Wrapper::instance();
  auto ret = deviceMallocManagedInternal(ptr, size, flags);
  assert(W->DeviceMemoryMap.count(ptr) == 0 &&
         "Expected non-duplicate entries in device memory");
  W->DeviceMemoryMap.insert({*ptr, size});
  DEBUG(for (auto &Entry
             : W->DeviceMemoryMap) std::cout
        << "MallocManaged Entry ptr " << Entry.first << " size " << Entry.second
        << "\n");

  return ret;
}

PREFIX(Error_t) PREFIX(Free)(void *devPtr) {
  Wrapper *W = Wrapper::instance();
  W->DeviceMemoryMap.erase(devPtr);
  return deviceFreeInternal(devPtr);
}
#endif

PREFIX(Error_t)
PREFIX(LaunchKernel)
(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
 PREFIX(Stream_t) stream) {
  Wrapper *W = Wrapper::instance();
  static std::size_t kernel_counter{0};
  PREFIX(Error_t) ret;

#if ENABLE_REPLAY_OPT
  auto PrintDims = [&](dim3 gridDim, dim3 blockDim) {
    printf("Grid (%u, %u, %u) Block (%u, %u, %u)\n", gridDim.x, gridDim.y,
           gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  };

  DEBUG(std::cout << "kernel " << W->SymbolTable[func] << "\n");
  DEBUG(std::cout << "kernel_counter " << kernel_counter << "\n");
  // assert(kernel_counter < sizeof(Configs) / sizeof(Configs[0]));
  DEBUG(std::cout << "Orig config "; PrintDims(gridDim, blockDim));
  DEBUG(std::cout << "New config ");
  DEBUG(PrintDims(Configs[kernel_counter].GridDim,
                  Configs[kernel_counter].BlockDim));
  DEBUG(printf(
      "%p %s Fixed Grid (%u, %u, %u) Block (%u, %u, %u)\n", func,
      W->SymbolTable[func].c_str(), Configs[kernel_counter].GridDim.x,
      Configs[kernel_counter].GridDim.y, Configs[kernel_counter].GridDim.z,
      Configs[kernel_counter].BlockDim.x, Configs[kernel_counter].BlockDim.y,
      Configs[kernel_counter].BlockDim.z));
  ret = deviceLaunchKernelInternal(func, Configs[kernel_counter].GridDim,
                                   Configs[kernel_counter].BlockDim, args,
                                   sharedMem, stream);

  // if (ret != PREFIX(Success)) {
  //   std::cout << "Failed to launch modified kernel: "
  //             << PREFIX(GetErrorString)(ret) << "\n";
  //   abort();
  // }
  kernel_counter++;
  return ret;
#endif

  CsvWriter *CsvW = CsvWriter::instance();

  DEBUG(printf("deviceLaunchKernel func %p name %s args %p sharedMem %zu\n",
               func, W->SymbolTable[func].c_str(), args, sharedMem);)
  int MaxThreadsPerBlock;
  auto Dim =
      blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z * gridDim.z;
  bool doTest = W->isTunable(func, MaxThreadsPerBlock, Dim);

  struct timeval t1, t2;

  if (doTest) {
    int WarpSize;
    int MultiProcessorCount;
    int MaxGridSizeX;
    W->getDeviceInfo(WarpSize, MultiProcessorCount, MaxGridSizeX);
    W->saveDeviceMemory();

    const char *EnvVarBlockSize = std::getenv("LIBREPLAY_BLOCKSIZE");
    std::string BlockSizeStr = (EnvVarBlockSize ? EnvVarBlockSize : "");
    if (!BlockSizeStr.empty()) {
      WarpSize = std::stoi(BlockSizeStr);
      MaxThreadsPerBlock = WarpSize;
    }

    // size_t BlockSize = 128;
    //  TODO: Use Apollo for exploration and tuning.
    for (int BlockSize = WarpSize; BlockSize <= MaxThreadsPerBlock;
         BlockSize *= 2) {
      int MaxGridDimSize = Dim / BlockSize + (Dim % BlockSize ? 1 : 0);
      int MaxGridSize = std::min(MaxGridDimSize, MaxGridSizeX);
      // TODO: fix GridSize not to past maximum Dim
      for (int GridSize = MultiProcessorCount; GridSize <= MaxGridSize;
           GridSize *= 2) {
        dim3 NewBlockDim, NewGridDim;
        NewBlockDim.x = BlockSize;
        NewGridDim.x = GridSize;
        // NewBlockDim = blockDim;

        gettimeofday(&t1, 0);
        ret = deviceLaunchKernelInternal(func, NewGridDim, NewBlockDim, args,
                                         sharedMem, stream);
        PREFIX(DeviceSynchronize)();
        gettimeofday(&t2, 0);
        W->restoreDeviceMemory();

        if (ret != PREFIX(Success)) {
          std::cout << "Failed to launch modified kernel: "
                    << PREFIX(GetErrorString)(ret) << "\n";
          abort();
        }
        double time =
            (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) /
            1000.0;

#if ENABLE_FAST_MODE
        W->updateTunings(func, Dim, NewGridDim, NewBlockDim, time);
#endif
        DEBUG(printf("%p %s Test Grid (%u, %u, %u) Block (%u, %u, %u) Dim "
                     "%u MaxGridSize %d %3.3f ms\n",
                     func, W->SymbolTable[func].c_str(), NewGridDim.x,
                     NewGridDim.y, NewGridDim.z, NewBlockDim.x, NewBlockDim.y,
                     NewBlockDim.z, Dim, MaxGridSize, time));
        CsvW->PrintCSVRow(kernel_counter, func, W->SymbolTable[func].c_str(),
                          NewGridDim, NewBlockDim, Dim, MaxGridSize, false,
                          time);
      }
    }

    W->freeCopies();
  }

#if ENABLE_FAST_MODE
  if (W->Tunings.count({func, Dim})) {
    CsvW->PrintCSVRow(kernel_counter, func, W->SymbolTable[func].c_str(),
                      W->Tunings[{func, Dim}].GridDim,
                      W->Tunings[{func, Dim}].BlockDim, Dim, 0, false,
                      W->Tunings[{func, Dim}].Time);
  }
#endif

  // Execute the original configuration.
  gettimeofday(&t1, 0);
  ret = deviceLaunchKernelInternal(func, gridDim, blockDim, args, sharedMem,
                                   stream);
  PREFIX(DeviceSynchronize)();
  gettimeofday(&t2, 0);

  double time =
      (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
  DEBUG(printf("%p %s Original Grid (%u, %u, %u) Block (%u, %u, %u) %3.3f ms\n",
               func, W->SymbolTable[func].c_str(), gridDim.x, gridDim.y,
               gridDim.z, blockDim.x, blockDim.y, blockDim.z, time));
  CsvW->PrintCSVRow(kernel_counter, func, W->SymbolTable[func].c_str(), gridDim,
                    blockDim, Dim, 0, true, time);

#if ENABLE_FAST_MODE
  W->updateTunings(func, Dim, gridDim, blockDim, time);
  auto &Measure = W->Tunings[{func, Dim}];
  OptWriter *OW = OptWriter::instance();
  OW->WriteOpt(Measure.GridDim, Measure.BlockDim);
#endif

  kernel_counter++;

  return ret;
}
}
