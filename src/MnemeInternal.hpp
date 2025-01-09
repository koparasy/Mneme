#ifdef ENABLE_HIP
#include "MnemeRecordHIP.hpp"
using MnemeRecordImplT = mneme::MnemeRecorderHIP;
#elif defined(ENABLE_CUDA)
#error Implementation is pending
#else
#error Neither ENABLE_HIP nor ENABLE_CUDA is defined
#endif
