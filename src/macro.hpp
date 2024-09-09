#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    auto err = CALL;                                                           \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errorName = nullptr;                                         \
      const char *errorString = nullptr;                                       \
      cuGetErrorName(err, &errorName);                                         \
      cuGetErrorString(err, &errorString);                                     \
      printf("ERROR @ %s:%d ->  %d\n ErrorName: %s\n, errorString:%s",         \
             __FILE__, __LINE__, err, errorName, errorString);                 \
      abort();                                                                 \
    }                                                                          \
  }

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(Twine(std::string{} + __FILE__ + ":" +                    \
                           std::to_string(__LINE__) + " => " + x))

#ifdef ENABLE_CUDA
#define DEVICE_FUNC(x) "cuda" x
#define PREFIX_UU(x) __cuda##x
#define PREFIX(x) cuda##x

namespace cuda {};

#elif defined(ENABLE_HIP)
#define DEVICE_FUNC(x) "hip" x
#define PREFIX_UU(x) __hip##x
#define PREFIX(x) hip##x
namespace hip {};
#endif

#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif
