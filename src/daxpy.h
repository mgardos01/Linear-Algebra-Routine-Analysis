#ifndef DAXPY_H
#define DAXPY_H

__global__ void daxpy_kernel(int n, double alpha, double* x, double* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = alpha * x[i] + y[i];
    }
}

float daxpy(int n, double alpha, double* x, double* y) {
    hipError_t err;

    const unsigned int blockSize = 64;
    const unsigned int gridSize = (n + blockSize + 1) / blockSize;

    double* d_x;
    hipMalloc(&d_x, n * sizeof(double));
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipMemcpy(d_x, x, n * sizeof(double), hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    double* d_y;
    hipMalloc(&d_y, n * sizeof(double));
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipMemcpy(d_y, y, n * sizeof(double), hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    hipEvent_t start, stop;
    float elapsed_ms;
    hipEventCreate(&start);
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);
    daxpy_kernel<<<gridSize, blockSize>>>(n, alpha, d_x, d_y);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    hipMemcpy(y, d_y, n * sizeof(double), hipMemcpyDeviceToHost);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_x);
    hipFree(d_y);

    return elapsed_ms;
}

float cublas_daxpy_wrapper(int n, double alpha, double* x, double* y) {
    hipError_t err;

    double* d_x;
    hipMalloc(&d_x, n * sizeof(double));
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipMemcpy(d_x, x, n * sizeof(double), hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    double* d_y;
    hipMalloc(&d_y, n * sizeof(double));
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipMemcpy(d_y, y, n * sizeof(double), hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    hipEvent_t start, stop;
    float elapsed_ms;
    hipEventCreate(&start);
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    hipMemcpy(y, d_y, n * sizeof(double), hipMemcpyDeviceToHost);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    cublasDestroy(handle);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_x);
    hipFree(d_y);

    return elapsed_ms;
}

#endif