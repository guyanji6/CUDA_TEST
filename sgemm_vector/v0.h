#include <cuda_runtime.h>
#include <iostream>
__global__ void sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // printf("Thread %d: Hello from the GPU!\n", i);
    if (row < M && col < N)
    {
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++)
        {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}