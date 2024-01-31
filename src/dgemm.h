// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix {
    int width;
    int height;
    int stride;
    double* elements;
};

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(
    Matrix A, int row, int col, double value, double alpha, double beta)
{
    A.elements[row * A.stride + col]
        = alpha * value + beta * A.elements[row * A.stride + col];
}

#define BLOCK_SIZE 32

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(
    Matrix A, Matrix B, Matrix C, const double alpha, const double beta)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue, alpha, beta);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
float dgemm(const unsigned int m, 
          const unsigned int n, 
          const unsigned int k,
          const double alpha,
          double* A, 
          double* B, 
          const double beta,  
          double* C)
{
    // A : m x k matrix
    // B : k x n matrix
    // C : m x n matrix
    // width represents cols
    // height represents rows

    hipError_t err;

    // Load A and B to device memory
    // std::cout << "STARTING GEMM FOR " << n << std::endl;
    size_t padded_m = (m % BLOCK_SIZE != 0 ? (floor(m / BLOCK_SIZE) + 1) * BLOCK_SIZE : m);
    // std::cout << "PADDED_M IS " << padded_m << std::endl;
    size_t padded_n = (n % BLOCK_SIZE != 0 ? (floor(n / BLOCK_SIZE) + 1) * BLOCK_SIZE : n);
    // std::cout << "PADDED_N IS " << padded_n << std::endl;
    size_t padded_k = (k % BLOCK_SIZE != 0 ? (floor(k / BLOCK_SIZE) + 1) * BLOCK_SIZE : k);
    // std::cout << "PADDED_K IS " << padded_k << std::endl;
    
    size_t d_pitch, s_pitch;

    // A : m x k matrix
    // std::cout << "ALLOCATING PADDED A" << std::endl;
    Matrix d_A;
    d_A.height = padded_m;
    d_A.width = padded_k;
    d_A.stride = padded_k;
    size_t A_size = padded_m * padded_k * sizeof(double);
    hipMalloc(&d_A.elements, A_size);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    size_t A_height = m;
    size_t A_width = k * sizeof(double);
    d_pitch = padded_k * sizeof(double);
    s_pitch = A_width;
    hipMemcpy2D(d_A.elements, d_pitch, A, s_pitch, A_width, A_height, hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    // B : k x n matrix
    // std::cout << "ALLOCATING PADDED B" << std::endl;
    Matrix d_B;
    d_B.height = padded_k;
    d_B.width = padded_n;
    d_B.stride = padded_n;
    size_t B_size = padded_k * padded_n * sizeof(double);
    hipMalloc(&d_B.elements, B_size);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    size_t B_height = k;
    size_t B_width = n * sizeof(double);
    d_pitch = padded_n * sizeof(double);
    s_pitch = B_width;
    hipMemcpy2D(d_B.elements, d_pitch, B, s_pitch, B_width, B_height, hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    // C : m x n matrix
    // std::cout << "ALLOCATING PADDED C" << std::endl;
    Matrix d_C;
    d_C.height = padded_m;
    d_C.width = padded_n;
    d_C.stride = padded_n;
    size_t C_size = padded_m * padded_n * sizeof(double);
    hipMalloc(&d_C.elements, C_size);
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    size_t C_height = m;
    size_t C_width = n * sizeof(double);
    d_pitch = padded_n * sizeof(double);
    s_pitch = C_width;
    hipMemcpy2D(d_C.elements, d_pitch, C, s_pitch, C_width, C_height, hipMemcpyHostToDevice);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    //////// Invoke kernel
    // std::cout << "INVOKING KERNEL" << std::endl;
    hipEvent_t start, stop;
    float elapsed_ms;
    hipEventCreate(&start);
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(d_B.width / dimBlock.x, d_A.height / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, alpha, beta);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    // Read C from device memory
    hipMemcpy2D(C, n * sizeof(double), d_C.elements, padded_n * sizeof(double), n * sizeof(double), m, hipMemcpyDeviceToHost);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    // Free device memory
    hipFree(d_A.elements);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipFree(d_B.elements);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }
    hipFree(d_C.elements);

    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("CUDA Error: %s: line %d\n", hipGetErrorString(err), __LINE__);
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return elapsed_ms;
}

float cublas_dgemm_wrapper(const unsigned int m, 
          const unsigned int n, 
          const unsigned int k,
          const double alpha,
          double* A, 
          double* B, 
          const double beta,  
          double* C) 
{
    // A : m x k matrix
    // B : k x n matrix
    // C : m x n matrix
    // std::cout << "STARTING CUBLAS DGEMM WRAPPER FOR " << n << std::endl;
    double* d_A;
    hipMalloc(&d_A, m * k * sizeof(double));
    hipMemcpy(d_A, A, m * k * sizeof(double), hipMemcpyHostToDevice);

    double* d_B;
    hipMalloc(&d_B, k * n * sizeof(double));
    hipMemcpy(d_B, B, k * n * sizeof(double), hipMemcpyHostToDevice);

    double* d_C;
    hipMalloc(&d_C, m * n * sizeof(double));
    hipMemcpy(d_C, C, m * n * sizeof(double), hipMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t op = CUBLAS_OP_N;

    hipEvent_t start, stop;
    float elapsed_ms;
    hipEventCreate(&start);
    hipEventCreate(&stop); 
    hipEventRecord(start, 0);
    cublasDgemm(handle, op, op, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);
    // std::cout << "cublasGemm ::: " << n << " took " << elapsed_ms << " ms\n";
    
    hipMemcpy(C, d_C, m * n * sizeof(double), hipMemcpyDeviceToHost);

    cublasDestroy(handle);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return elapsed_ms;
}