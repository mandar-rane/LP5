#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to add two vectors
__global__
void addVectors(float *a, float *b, float *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Size of vectors
    int size = 1000000;

    // Host vectors
    float *h_a, *h_b, *h_c;

    // Device vectors
    float *d_a, *d_b, *d_c;

    // Allocate memory for host vectors
    h_a = new float[size];
    h_b = new float[size];
    h_c = new float[size];

    // Initialize host vectors
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch kernel
    addVectors<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
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
