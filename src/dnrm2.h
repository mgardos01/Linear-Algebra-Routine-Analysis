#ifndef NRM2_H
#define NRM2_H

#include "status.h"
#include <cmath>
#include <iostream>

__device__ double fabspow2(double num)
{
    return pow(fabs(num), 2);
}

__global__ void fabspow2_kernel(double* x, unsigned int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = fabspow2(x[i]);
    }
}

template <unsigned int blockSize>
__device__ void sum_warpReduce(volatile double* sdata, unsigned int tid)
{
    if (blockSize >= 64) {
        sdata[tid] += sdata[tid + 32];
    }
    if (blockSize >= 32) {
        sdata[tid] += sdata[tid + 16];
    }
    if (blockSize >= 16) {
        sdata[tid] += sdata[tid + 8];
    }
    if (blockSize >= 8) {
        sdata[tid] += sdata[tid + 4];
    }
    if (blockSize >= 4) {
        sdata[tid] += sdata[tid + 2];
    }
    if (blockSize >= 2) {
        sdata[tid] += sdata[tid + 1];
    }
}

template <unsigned int blockSize>
__global__ void sum_kernel(double* g_idata, double* g_odata, unsigned int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        sum_warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

float dnrm2(const double* x, double* y, unsigned int n)
{
    const unsigned int blockSize = 64;
    const unsigned int gridSize = 128;

    hipEvent_t start, stop;
    float elapsed_ms;

    double* d_in;
    hipMalloc(&d_in, n * sizeof(double));
    hipMemcpy(d_in, x, n * sizeof(double), hipMemcpyHostToDevice);

    double* d_out;
    hipMalloc(&d_out, n * sizeof(double));

    dim3 dimBlock(blockSize);
    dim3 dimGrid(gridSize);
    // dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    fabspow2_kernel<<<dimGrid, dimBlock>>>(d_in, n);
    sum_kernel<blockSize><<<dimGrid, dimBlock, blockSize * sizeof(double)>>>(d_in, d_out, n);
    sum_kernel<blockSize><<<1, dimBlock, blockSize * sizeof(double)>>>(d_out, d_out, n);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);
    hipMemcpy(y, d_out, sizeof(double), hipMemcpyDeviceToHost);

    y[0] = sqrt(y[0]);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_in);
    hipFree(d_out);

    return elapsed_ms;
}

float cublas_dnrm2_wrapper(const double* x, double* y, unsigned int n) {
    hipEvent_t start, stop;
    float elapsed_ms;

    double* d_in;
    hipMalloc(&d_in, n * sizeof(double));
    hipMemcpy(d_in, x, n * sizeof(double), hipMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    cublasDnrm2(handle, n, d_in, 1, y);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    cublasDestroy(handle);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_in);

    return elapsed_ms;
}

#endif