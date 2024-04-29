#include <iostream>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel for matrix multiplication
__global__
void matrixMul(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

int main() {
    // Host matrices
    int *h_a, *h_b, *h_c;

    // Device matrices
    int *d_a, *d_b, *d_c;

    // Allocate memory for host matrices
    h_a = new int[N * N];
    h_b = new int[N * N];
    h_c = new int[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output results
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
