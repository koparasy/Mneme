#include "MnemeRecordHIP.hpp"
#include <thread>

using namespace mneme;

MnemeRecorderHIP &MnemeRecorderHIP::instance() {
  static MnemeRecorderHIP Recorder{};
  return Recorder;
}

extern "C" {
void __hipRegisterFatBinaryEnd(void *ptr) {
  auto mneme = MnemeRecorderHIP::instance();
  mneme.registerFatBinEnd(ptr);
}

void **__hipRegisterFatBinary(void *fatbin) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.registerFatBin(fatbin);
}

void __hipRegisterVar(void **fatbinHandle, char *hostVar, char *deviceAddress,
                      const char *deviceName, int ext, size_t size,
                      int constant, int global) {
  auto mneme = MnemeRecorderHIP::instance();
  mneme.registerVar(fatbinHandle, hostVar, deviceAddress, deviceName, ext, size,
                    constant, global);
};

void __hipRegisterFunction(void **fatbinHandle, const char *hostFun,
                           char *deviceFun, const char *deviceName,
                           int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim,
                           dim3 *gDim, int *wSize) {
  auto mneme = MnemeRecorderHIP::instance();
  mneme.registerFunc(fatbinHandle, hostFun, deviceFun, deviceName, thread_limit,
                     tid, bid, bDim, gDim, wSize);
};

hipError_t hipMalloc(void **ptr, size_t size) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtMalloc(ptr, size);
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtManagedMalloc(ptr, size, flags);
};

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtHostMalloc(ptr, size, flags);
}

hipError_t hipFree(void *ptr) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtFree(ptr);
};

hipError_t hipHostFree(void *ptr) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtHostFree(ptr);
}

hipError_t hipLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                           void **args, size_t sharedMem, hipStream_t stream) {
  auto mneme = MnemeRecorderHIP::instance();
  return mneme.rtLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
}
