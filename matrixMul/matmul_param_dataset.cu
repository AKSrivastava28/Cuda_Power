// matmul_param_dataset.cu
// Corrected matrix multiply supporting:
//  - WORK_PER_THREAD (each thread computes consecutive columns)
//  - STRIDE_ACCESS (k-loop increment)
//  - USE_SHARED (0/1): shared tile of A only (safe, robust)
//  - USE_UNROLL (0/1): optional loop unrolling (safe)
// Prints parseable metrics for the Python driver.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#ifndef WORK_PER_THREAD
#define WORK_PER_THREAD 1
#endif
#ifndef GRID_SCALE_X
#define GRID_SCALE_X 1.0f
#endif
#ifndef GRID_SCALE_Y
#define GRID_SCALE_Y 1.0f
#endif
#ifndef USE_SHARED
#define USE_SHARED 0
#endif
#ifndef USE_UNROLL
#define USE_UNROLL 0
#endif
#ifndef STRIDE_ACCESS
#define STRIDE_ACCESS 1
#endif

// Kernel: each thread computes WORK_PER_THREAD consecutive output columns
__global__ void MatrixMulCUDA(float *C, const float *A, const float *B, int M, int K, int N)
{
    // M = dimsA_y (rows of A)
    // K = dimsA_x = dimsB_y (inner dimension)
    // N = dimsB_x (cols of B)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    // block covers BLOCK_SIZE * WORK_PER_THREAD columns
    int block_cols = BLOCK_SIZE * WORK_PER_THREAD;
    int block_start_col = bx * block_cols;
    int base_col = block_start_col + tx * WORK_PER_THREAD; // this thread's first column

#if USE_SHARED
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; // tile of A (K-direction tile width is BLOCK_SIZE)
#endif

    // Boundary check for row
    if (row >= M) return;

    // compute accumulation for each of the WORK_PER_THREAD outputs
    float acc[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w) acc[w] = 0.0f;

    // iterate over K in tiles of BLOCK_SIZE
    for (int tile = 0; tile < K; tile += BLOCK_SIZE) {
        int tile_width = min(BLOCK_SIZE, K - tile);

        // Load A tile into shared memory (if enabled)
#if USE_SHARED
        if (tx < tile_width) {
            // Each thread loads one element of the tile row
            As[ty][tx] = A[(size_t)row * (size_t)K + (tile + tx)];
        }
        // For threads where tx >= tile_width, As entry is unused
        __syncthreads();
#endif

        // For k within this tile, step by STRIDE_ACCESS
#if USE_UNROLL
#pragma unroll
#endif
        for (int krel = 0; krel < tile_width; krel += STRIDE_ACCESS) {
            int k = tile + krel;
            // fetch A value (from shared if enabled, else directly)
            float a_val;
#if USE_SHARED
            a_val = As[ty][krel];
#else
            a_val = A[(size_t)row * (size_t)K + k];
#endif
            // for each output computed by this thread
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                int col = base_col + w;
                if (col >= N) continue; // boundary check

                // B element: row = k, col = col
                float b_val = B[(size_t)k * (size_t)N + col];
                acc[w] += a_val * b_val;
            }
        } // end k in tile

#if USE_SHARED
        __syncthreads();
#endif
    } // end tiles

    // Write results to C (with boundary checks)
    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        int col = base_col + w;
        if (col < N) {
            C[(size_t)row * (size_t)N + col] = acc[w];
        }
    }
}

// Host helpers
void ConstantInit(float *data, size_t size, float val)
{
    for (size_t i = 0; i < size; ++i) data[i] = val;
}

int main()
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    printf("CONFIG: BLOCK_SIZE=%d WORK_PER_THREAD=%d GRID_SCALE_X=%.2f GRID_SCALE_Y=%.2f USE_SHARED=%d USE_UNROLL=%d STRIDE_ACCESS=%d\n",
           BLOCK_SIZE, WORK_PER_THREAD, GRID_SCALE_X, GRID_SCALE_Y, USE_SHARED, USE_UNROLL, STRIDE_ACCESS);

    // Dimensions following your pattern
    int K = 5 * 2 * BLOCK_SIZE;                 // inner dim (width of A, height of B)
    int M = 5 * 2 * BLOCK_SIZE;                 // rows of A
    int N = 5 * 4 * BLOCK_SIZE * WORK_PER_THREAD; // cols of B

    // sanity
    if (K <= 0 || M <= 0 || N <= 0) {
        printf("ERROR: invalid matrix dims\n");
        return 1;
    }

    size_t sizeA = (size_t)M * (size_t)K;
    size_t sizeB = (size_t)K * (size_t)N;
    size_t sizeC = (size_t)M * (size_t)N;

    size_t memA = sizeof(float) * sizeA;
    size_t memB = sizeof(float) * sizeB;
    size_t memC = sizeof(float) * sizeC;

    float *h_A = (float*)malloc(memA);
    float *h_B = (float*)malloc(memB);
    float *h_C = (float*)malloc(memC);
    if (!h_A || !h_B || !h_C) {
        printf("ERROR: host malloc failed\n");
        return 1;
    }

    ConstantInit(h_A, sizeA, 1.0f);
    ConstantInit(h_B, sizeB, 0.01f);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, memA); if (err != cudaSuccess) { printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc((void**)&d_B, memB); if (err != cudaSuccess) { printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); return 1; }
    err = cudaMalloc((void**)&d_C, memC); if (err != cudaSuccess) { printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); cudaFree(d_B); return 1; }

    err = cudaMemcpy(d_A, h_A, memA, cudaMemcpyHostToDevice); if (err != cudaSuccess) { printf("cudaMemcpy A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMemcpy(d_B, h_B, memB, cudaMemcpyHostToDevice); if (err != cudaSuccess) { printf("cudaMemcpy B failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Grid calculation: each block covers BLOCK_SIZE rows and BLOCK_SIZE * WORK_PER_THREAD columns.
    int block_cols = BLOCK_SIZE * WORK_PER_THREAD;
    int grid_x = (N + block_cols - 1) / block_cols;
    int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Optionally scale grid (keeps correctness because we still bound-check on writes; but scaling >1 just creates more blocks with empty writes)
    grid_x = (int)ceilf(grid_x * GRID_SCALE_X);
    grid_y = (int)ceilf(grid_y * GRID_SCALE_Y);
    if (grid_x < 1) grid_x = 1;
    if (grid_y < 1) grid_y = 1;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_x, grid_y);

    int threads_per_block = threads.x * threads.y;
    printf("THREADS_PER_BLOCK: %d\n", threads_per_block);
    printf("STRIDE_ACCESS: %d\n", STRIDE_ACCESS);

    // Warmup launch & error check
    MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, M, K, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch (warmup) error: %s\n", cudaGetErrorString(err));
        // cleanup
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA sync (warmup) error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int nIter = 300;
    cudaEventRecord(start, 0);
    for (int i = 0; i < nIter; ++i) {
        MatrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, M, K, N);
    }
    // check launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch (timed) error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) {
        printf("CUDA event record error: %s\n", cudaGetErrorString(err));
    }
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("CUDA sync (timed) error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msecPerMatrixMul = msecTotal / (float)nIter;
    // compute flops: 2 * M * K * N
    double flops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (flops * 1e-9) / (msecPerMatrixMul / 1000.0);

    printf("GFLOPS: %.6f\n", gflops);
    printf("MSEC_PER_MATMUL: %.6f\n", msecPerMatrixMul);

    // Copy back and verify result
    err = cudaMemcpy(h_C, d_C, memC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy back failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }

    bool correct = true;
    // reference value: each element should be sum_k A_row_k * B_k_col
    // since A entries = 1.0 and B entries = 0.01, expected value = K * 0.01
    double ref = (double)K * 0.01;
    for (size_t idx = 0; idx < sizeC; ++idx) {
        double v = (double)h_C[idx];
        double abs_err = fabs(v - ref);
        if (abs_err > 1e-3) { correct = false; break; }
    }

    printf("RESULT: %s\n", correct ? "PASS" : "FAIL");

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return correct ? 0 : 1;
<<<<<<< HEAD
}
=======
}
>>>>>>> 4cfb83f (Apply local changes: remove deleted files, add new files, update .gitignore)
