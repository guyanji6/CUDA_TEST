#include <cuda_runtime.h>
#include <iostream>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define BLOCK_SIZE 32
#define NUM_PER_THREAD 4
__global__ void sgemm0(const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // if(row == 0){
    //     printf("Thread %d: Hello from the GPU!\n", col);
    // }
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

__global__ void sgemm1(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N && col + 3 < N)
    {
        float sum[4] = {0.0f};
        for (size_t k = 0; k < K; k ++)
        {
            float a = A[row * K + k];
            float4 b = reinterpret_cast<float4 *>(B + k * N)[col];
            sum[0] += a * b.x;
            sum[1] += a * b.y;
            sum[2] += a * b.z;
            sum[3] += a * b.w;
        }
        reinterpret_cast<float4 *>(C + row * N)[col] = make_float4(sum[0], sum[1], sum[2], sum[3]);
    }
}