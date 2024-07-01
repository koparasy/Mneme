#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    auto err = CALL;                                                           \
    if (err != CUDA_SUCCESS) {                                                 \
      printf("ERROR @ %s:%d ->  %d\n", __FILE__, __LINE__, err);               \
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
