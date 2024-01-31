#ifndef STATUS_H
#define STATUS_H

#include <hip/hip_runtime.h>

struct libStatus_t {
    hipError_t hip_error;
    float elapsed_time;
    libStatus_t(hipError_t hip_error, float elapsed_time)
    {
        hip_error = hip_error;
        elapsed_time = elapsed_time;
    }
    libStatus_t(hipError_t hip_error)
    {
        hip_error = hip_error;
        elapsed_time = -1;
    }
};

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void
gpuAssert(hipError_t code, const char* file, int line /*, bool abort = true*/)
{
    if (code != hipSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        // if (abort)
        //     exit(EXIT_FAILURE);
    }
}

#endif