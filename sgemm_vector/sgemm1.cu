#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <fstream>
#include "kernel.h"

void sgemmCPU(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
bool compare(std::vector<float> h_C, std::vector<float> h_C_CPU, int M, int N)
{
    bool isCorrect = true;
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5)
        {
            std::cerr << "Mismatch at element " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_CPU[i] << std::endl;
            isCorrect = false;
            break;
        }
    }
    return isCorrect;
}

void WriteMatrix(std::string name, std::vector<float> h_C, int M, int N)
{
    std::ofstream outFile(name);
    if (!outFile.is_open())
    {
        std::cerr << "Unable to open file: " << name << std::endl;
    }
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            outFile << h_C[i * N + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
}

int main()
{
    cudaSetDevice(0);
    // 矩阵维度
    int M = 2048; // A 的行数
    int N = 1024; // B 的列数
    int K = 256;  // A 的列数和 B 的行数

    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_CPU(M * N);
    std::vector<float> h_C_cublas(M * N);
    cudaEvent_t start, stop;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = static_cast<float>(i % 10);
    }
    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = static_cast<float>(i % 10);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C, *d_C_cublas;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    cudaMalloc((void **)&d_C_cublas, M * N * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块大小和网格大小
    int threadsPerBlock = 32;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock/4, (M + threadsPerBlock - 1) / threadsPerBlock);

    sgemm1<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(start);
    sgemm1<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 将结果从设备复制回主机
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 在 CPU 上计算矩阵乘法
    sgemmCPU(h_A.data(), h_B.data(), h_C_CPU.data(), M, N, K);

    // 比较 GPU 和 CPU 的结果
    bool isCorrect = compare(h_C, h_C_CPU, M, N);
    // WriteMatrix("cpu res.txt", h_C_CPU, M, N);
    WriteMatrix("kernel res.txt", h_C, M, N);
    // 使用 cuBLAS 计算矩阵乘法
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N,
                d_A, K,
                &beta, d_C_cublas, N);
    cudaEventRecord(start1);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N,
                d_A, K,
                &beta, d_C_cublas, N);
    cudaEventRecord(stop1);
    cudaDeviceSynchronize();
    // cudaEventSynchronize(stop1);
    // milliseconds = 1.1;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    std::cout << "cublas execution time: " << milliseconds << " ms" << std::endl;

    // cudaMemcpy(h_C_cublas.data(), d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // bool isCorrect2 = compare(h_C_cublas, h_C_CPU, M, N);
    // WriteMatrix("cublas res.txt", h_C_cublas, M, N);
    if (isCorrect)
    {
        std::cout << "Matrix multiplication pass!" << std::endl;
    }
    else
    {
        std::cerr << "Matrix multiplication mismatch" << std::endl;
    }
    // if (isCorrect2)
    // {
    //     std::cout << "cublas multiplication pass!" << std::endl;
    // }
    // else
    // {
    //     std::cerr << "cublas multiplication mismatch" << std::endl;
    // }
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}