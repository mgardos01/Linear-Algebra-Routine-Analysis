#ifndef ASUM_H
#define ASUM_H

#include "status.h"
#include <iostream>

/** 
 * @brief warp reduce device kernel for absolute sum kernel
 * 
 * @tparam blockSize 
 * @param sdata shared data array
 * @param tid thread id
 */
template <unsigned int blockSize>
__device__ void dasum_warpReduce(volatile double* sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += fabs(sdata[tid + 32]);
    if (blockSize >= 32)
        sdata[tid] += fabs(sdata[tid + 16]);
    if (blockSize >= 16)
        sdata[tid] += fabs(sdata[tid + 8]);
    if (blockSize >= 8)
        sdata[tid] += fabs(sdata[tid + 4]);
    if (blockSize >= 4)
        sdata[tid] += fabs(sdata[tid + 2]);
    if (blockSize >= 2)
        sdata[tid] += fabs(sdata[tid + 1]);
}

/** 
 * @brief Kernel for absolute summation of g_idata array into g_odata
 * 
 * @tparam blockSize 
 * @param g_idata input data array 
 * @param g_odata output data array
 * @param n size of array
 */
template <unsigned int blockSize>
__global__ void dasum_kernel(double* g_idata, double* g_odata, unsigned int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += fabs(g_idata[i]) + fabs(g_idata[i + blockSize]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += fabs(sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += fabs(sdata[tid + 128]);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += fabs(sdata[tid + 64]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        dasum_warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/** 
 * @brief abstraction for absolute sum kernel
 * 
 * @param x input array
 * @param y output array
 * @param n size of array
 * @return float elapsed time of the function
 */
float dasum(const double* x, double* y, unsigned int n)
{
    hipEvent_t start, stop;
    float elapsed_ms;
    const unsigned int blockSize = 128;
 
    double* d_in;
    hipMalloc(&d_in, n * sizeof(double));
    hipMemcpy(d_in, x, n * sizeof(double), hipMemcpyHostToDevice);

    double* d_out;
    hipMalloc(&d_out, n * sizeof(double));

    dim3 dimBlock(blockSize);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    dasum_kernel<blockSize><<<dimGrid, dimBlock, blockSize * sizeof(double)>>>(d_in, d_out, n);
    dasum_kernel<blockSize><<<1, dimBlock, blockSize * sizeof(double)>>>(d_out, d_out, n);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    hipMemcpy(y, d_out, sizeof(double), hipMemcpyDeviceToHost);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_in);
    hipFree(d_out);

    return elapsed_ms;
}

/** 
 * @brief Wrapper for cublasDasum
 * 
 * @param x input array
 * @param y output array (singular)
 * @param n size of x
 * @return elapsed time in ms 
 */
float cublas_dasum_wrapper(const double* x, double* y, unsigned int n) {
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

    cublasDasum(handle, n, d_in, 1, y);

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