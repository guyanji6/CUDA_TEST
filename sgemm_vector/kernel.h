#include <cuda_runtime.h>
#include <iostream>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define BLOCK_SIZE 32
#define NUM_PER_THREAD 4
#define TILE_SIZE 4
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
/* float4 col*/
__global__ void sgemm1(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N && col + 3 < N)
    {
        float sum[4] = {0.0f};
        for (size_t k = 0; k < K; k++)
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

/* shared */
template <int M, int N, int K>
__global__ void sgemm2(float *A, float *B, float *C)
{
    __shared__ float smemA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float smemB[BLOCK_SIZE][BLOCK_SIZE];
    int BK = BLOCK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;

    for (size_t out_k = 0; out_k < K; out_k += BK)
    {
        if (row < M && (out_k + tx) < K)
            smemA[ty][tx] = A[row * K + out_k + tx];
        else
            smemA[ty][tx] = 0.0f;
        if ((out_k + ty) < K && col < N)
            smemB[ty][tx] = B[(out_k + ty) * N + col];
        else
            smemB[ty][tx] = 0.0f;
        __syncthreads();
        for (size_t in_k = 0; in_k < BK; in_k += 1)
        {
            sum += smemA[ty][in_k] * smemB[in_k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

/* shared , each thread compute a tile */
template <int M, int N, int K>
__global__ void sgemm3(float *A, float *B, float *C)
{
    __shared__ float smemA[BLOCK_SIZE * TILE_SIZE][BLOCK_SIZE];
    __shared__ float smemB[BLOCK_SIZE][BLOCK_SIZE * TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * BLOCK_SIZE * TILE_SIZE + ty * TILE_SIZE;
    int col = bx * BLOCK_SIZE * TILE_SIZE + tx * TILE_SIZE;
    float sum[TILE_SIZE][TILE_SIZE] = {{0.0f}};

    for (int k_offset = 0; k_offset < K; k_offset += BLOCK_SIZE)
    {
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
        {
            int load_row = row + i;
            int load_col = k_offset + tx;
            if (load_row < M && load_col < K)
            {
                // 索引：行=ty*TILE_SIZE+i, 列=tx
                smemA[ty * TILE_SIZE + i][tx] = A[load_row * K + load_col];
            }
            else
            {
                smemA[ty * TILE_SIZE + i][tx] = 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < TILE_SIZE; j++)
        {
            int load_row = k_offset + ty; // K维度偏移
            int load_col = col + j;
            if (load_row < K && load_col < N)
            {
                // 行=ty, 列=tx*TILE_SIZE+j, 列方向*TILE_SIZE
                smemB[ty][tx * TILE_SIZE + j] = B[load_row * N + load_col];
            }
            else
            {
                smemB[ty][tx * TILE_SIZE + j] = 0.0f;
            }
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            // 预取A的TILE_SIZE个元素到寄存器
            float a_reg[TILE_SIZE];
#pragma unroll
            for (int i = 0; i < TILE_SIZE; i++)
            {

                a_reg[i] = smemA[ty * TILE_SIZE + i][k];
            }

            // 预取B的TILE_SIZE个元素到寄存器
            float b_reg[TILE_SIZE];
#pragma unroll
            for (int j = 0; j < TILE_SIZE; j++)
            {
                b_reg[j] = smemB[k][tx * TILE_SIZE + j];
            }

// 寄存器级矩阵乘法
#pragma unroll
            for (int i = 0; i < TILE_SIZE; i++)
            {
#pragma unroll
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TILE_SIZE; i++)
    {
#pragma unroll
        for (int j = 0; j < TILE_SIZE; j++)
        {
            if ((row + i) < M && (col + j) < N)
            {
                C[(row + i) * N + col + j] = sum[i][j];
            }
        }
    }
}