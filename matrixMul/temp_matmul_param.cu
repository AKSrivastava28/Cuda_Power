// matmul_param_dataset.cu
// Configurable CUDA matrix multiply that prints parseable outputs for profiling.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef WORK_PER_THREAD
#define WORK_PER_THREAD 4
#endif
#ifndef GRID_SCALE_X
#define GRID_SCALE_X 2.0f
#endif
#ifndef GRID_SCALE_Y
#define GRID_SCALE_Y 2.0f
#endif
#ifndef USE_SHARED
#define USE_SHARED 1
#endif
#ifndef USE_UNROLL
#define USE_UNROLL 1
#endif

__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;

#if USE_SHARED
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
#endif

    for (int wp = 0; wp < WORK_PER_THREAD; ++wp)
    {
        int col = (bx * BLOCK_SIZE + tx) + wp * BLOCK_SIZE;
        float Cvalue = 0.0f;

#if USE_UNROLL
#pragma unroll
#endif
        for (int k = 0; k < wA; k++)
        {
#if USE_SHARED
            int tiledRow = ty;
            int tiledCol = tx;
            As[tiledRow][tiledCol] = A[(by * BLOCK_SIZE + tiledRow) * wA + (k)];
            Bs[tiledRow][tiledCol] = B[(k) * wB + (bx * BLOCK_SIZE +
tiledCol) + wp * BLOCK_SIZE];
            __syncthreads();
            float a = As[tiledRow][k % BLOCK_SIZE];
            float b = Bs[k % BLOCK_SIZE][tiledCol];
            Cvalue += a * b;
            __syncthreads();
#else
            float a = A[row * wA + k];
            float b = B[k * wB + col];
            Cvalue += a * b;
#endif
        }
        C[row * wB + col] = Cvalue;
    }
}

void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
        data[i] = val;
}

int main()
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    printf("CONFIG: BLOCK_SIZE=%d WORK_PER_THREAD=%d GRID_SCALE_X=%.2f GRID_SCALE_Y=%.2f USE_SHARED=%d USE_UNROLL=%d\n", BLOCK_SIZE, WORK_PER_THREAD, GRID_SCALE_X, GRID_SCALE_Y, USE_SHARED, USE_UNROLL);

    int dimsA_x = 5 * 2 * BLOCK_SIZE;
    int dimsA_y = 5 * 2 * BLOCK_SIZE;
    int dimsB_x = 5 * 4 * BLOCK_SIZE * WORK_PER_THREAD;
    int dimsB_y = 5 * 2 * BLOCK_SIZE;

    if (dimsA_x != dimsB_y)
    {
        printf("ERROR: Matrix dimensions mismatch: dimsA_x(%d) !=dimsB_y(%d)\n", dimsA_x, dimsB_y);
        return 1;
    }

    unsigned int size_A = dimsA_x * dimsA_y;
    unsigned int size_B = dimsB_x * dimsB_y;
    unsigned int size_C = dimsB_x * dimsA_y;
    size_t mem_size_A = sizeof(float) * (size_t)size_A;
    size_t mem_size_B = sizeof(float) * (size_t)size_B;
    size_t mem_size_C = sizeof(float) * (size_t)size_C;

    float *h_A = (float *)malloc(mem_size_A);
    float *h_B = (float *)malloc(mem_size_B);
    float *h_C = (float *)malloc(mem_size_C);
    if (!h_A || !h_B || !h_C)
    {
        printf("ERROR: host malloc failed\n");
        return 1;
    }

    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, 0.01f);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;
    err = cudaMalloc((void **)&d_A, mem_size_A);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_B, mem_size_B);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_C, mem_size_C);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int grid_x = (int)((dimsB_x + threads.x - 1) / threads.x * GRID_SCALE_X);
    int grid_y = (int)((dimsA_y + threads.y - 1) / threads.y * GRID_SCALE_Y);
    dim3 grid(grid_x, grid_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, dimsA_x, dimsB_x);
    cudaDeviceSynchronize();

    int nIter = 300;
    cudaEventRecord(start);
    for (int j = 0; j < nIter; ++j)
    {
        MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, dimsA_x, dimsB_x);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msecPerMatrixMul = msecTotal / (float)nIter;
    double flops = 2.0 * (double)dimsA_x * (double)dimsA_y * (double)dimsB_x;
    double gflops = (flops * 1e-9) / (msecPerMatrixMul / 1000.0);

    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (unsigned int i = 0; i < size_C; ++i)
    {
        double abs_err = fabs(h_C[i] - (dimsA_x * 0.01f));
        if (abs_err > 1e-3)
        {
            correct = false;
            break;
        }
    }

    int threads_per_block = threads.x * threads.y;
    printf("GFLOPS: %.6f\n", gflops);
    printf("MSEC_PER_MATMUL: %.6f\n", msecPerMatrixMul);
    printf("THREADS_PER_BLOCK: %d\n", threads_per_block);
    printf("RESULT: %s\n", correct ? "PASS" : "FAIL");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return correct ? 0 : 1;
}

