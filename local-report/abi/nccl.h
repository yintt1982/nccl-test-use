#ifndef NCCL_ABI_H_
#define NCCL_ABI_H_
#ifdef _WIN32
#define NCCL_OS_WINDOWS 1
typedef unsigned long pid_t;
#endif
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7
} ncclResult_t;
#endif
