#include "MnemeMemoryHIP.hpp"
#include "MnemeRecord.hpp"
#include <dlfcn.h>

namespace mneme {

class MnemeRecorderHIP;
template <> struct DeviceTraits<MnemeRecorderHIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using AllocGranularityFlags = hipMemAllocationGranularity_flags;
};

class MnemeRecorderHIP
    : public MnemeRecorder<MnemeRecorderHIP, MnemeMemoryBlobHIP> {
private:
  MnemeRecorderHIP() = default;

public:
  static auto *getRTLib() { return dlopen("libamdhip64.so", RTLD_NOW); }
  static const char *getLaunchKernelFnName() { return "hipLaunchKernel"; }
  static const char *getDeviceMallocFnName() { return "hipMalloc"; }
  static const char *getPinnedMallocFnName() { return "hipHostMalloc"; }
  static const char *getManagedMallocFnName() { return "hipMallocManaged"; }
  static const char *getDeviceFreeFnName() { return "hipFree"; }
  static const char *getPinnedFreeFnName() { return "hipHostFree"; }
  static const char *getUURegisterFunctionFnName() {
    return "__hipRegisterFunction";
  }
  static const char *getUURegisterVarFnName() { return "__hipRegisterVar"; }
  static const char *getUURegisterFatbinFnName() {
    return "__hipRegisterFatBinary";
  }

  static constexpr bool hasFatBinEnd = false;

  static MnemeRecorderHIP &instance();

  void extractIR();
  MnemeRecorderHIP(MnemeRecorderHIP &) = delete;
  MnemeRecorderHIP(MnemeRecorderHIP &&) = delete;
};
} // namespace mneme
