#include <cuda_runtime.h>
#include "filters.cuh"
#include "image.cuh"
#include "kernels.cuh"
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define CHANNELS 3

void gaussianBlurCUDAoptimize(Image* src, Image* dst, int radius, float sigma) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Calculate Gaussian kernel on CPU
    int kernelSize = 2 * radius + 1;
    float* kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float sum = 0.0f;
    
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            float value = exp(-(x*x + y*y)/(2*sigma*sigma));
            kernel[(y+radius) * kernelSize + (x+radius)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for(int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Allocate device memory
    unsigned char *d_src, *d_dst;
    float *d_kernel;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    size_t kernel_size = kernelSize * kernelSize * sizeof(float);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMalloc(&d_kernel, kernel_size);
    
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const size_t smem_size = (kernelSize * kernelSize * sizeof(float)) + 
                            (tile_w * tile_h * src->channels * sizeof(unsigned char));
    
    // Check shared memory size against device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%lu bytes) exceeds device limit (%lu bytes)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }
    
    // Set up grid and block dimensions
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (src->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (src->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    // Launch optimized kernel
    gaussianKernelOptimized<<<grid, block, smem_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius,
        d_kernel
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    free(kernel);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void gaussianBlurCUDA(Image* src, Image* dst, int radius, float sigma) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Calculate Gaussian kernel on CPU
    int kernelSize = 2 * radius + 1;
    float* kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float sum = 0.0f;
    
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            float value = exp(-(x*x + y*y)/(2*sigma*sigma));
            kernel[(y+radius) * kernelSize + (x+radius)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for(int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Allocate device memory
    unsigned char *d_src, *d_dst;
    float *d_kernel;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    size_t kernel_size = kernelSize * kernelSize * sizeof(float);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMalloc(&d_kernel, kernel_size);
    
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    gaussianKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius,
        sigma,
        d_kernel
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    free(kernel);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void embossCUDAoptimize(Image* src, Image* dst, float intensity) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size
    const int radius = 1;
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const size_t smem_size = tile_w * tile_h * src->channels * sizeof(unsigned char);
    
    // Check shared memory size against device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%lu bytes) exceeds device limit (%lu bytes)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }
    
    // Set up grid and block dimensions
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (src->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (src->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    // Launch kernel with dynamic shared memory size
    embossKernelOptimized<<<grid, block, smem_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        intensity
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void embossCUDA(Image* src, Image* dst, float intensity) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    embossKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        intensity
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void erosionCUDAoptimize(Image* src, Image* dst, int radius) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size based on actual radius
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const size_t smem_size = tile_w * tile_h * src->channels * sizeof(unsigned char);
    
    // Check if shared memory size exceeds device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%lu bytes) exceeds device limit (%lu bytes)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }
    
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (src->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (src->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    // Launch kernel with dynamic shared memory size
    erosionKernelOptimized<<<grid, block, smem_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void erosionCUDA(Image* src, Image* dst, int radius) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    erosionKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void dilationCUDAoptimize(Image* src, Image* dst, int radius) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size based on actual radius
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const size_t smem_size = tile_w * tile_h * src->channels * sizeof(unsigned char);
    
    // Check if shared memory size exceeds device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%lu bytes) exceeds device limit (%lu bytes)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }
    
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (src->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (src->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    // Launch kernel with dynamic shared memory size
    dilationKernelOptimized<<<grid, block, smem_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void dilationCUDA(Image* src, Image* dst, int radius) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    dilationKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void waveCUDA(Image* src, Image* dst, float amplitudeX, float amplitudeY, float frequencyX, float frequencyY) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    waveKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        amplitudeX,
        amplitudeY,
        frequencyX,
        frequencyY
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void oilPaintingCUDAoptimize(Image* src, Image* dst, int radius, int intensityLevels) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    // Calculate shared memory size based on actual radius
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const size_t smem_size = tile_w * tile_h * src->channels * sizeof(unsigned char);
    
    // Check if shared memory size exceeds device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%lu bytes) exceeds device limit (%lu bytes)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }
    
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (src->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (src->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    // Launch kernel with dynamic shared memory size
    oilPaintingKernelOptimized<<<grid, block, smem_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius,
        intensityLevels
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

void oilPaintingCUDA(Image* src, Image* dst, int radius, int intensityLevels) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    unsigned char *d_src, *d_dst;
    size_t size = src->width * src->height * src->channels * sizeof(unsigned char);
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
    
    int total_elements = src->width * src->height * src->channels;
    int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    oilPaintingKernel<<<num_blocks, block_size>>>(
        d_src,
        d_dst,
        src->width,
        src->height,
        src->channels,
        radius,
        intensityLevels
    );
    
    cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %.4f seconds\n", milliseconds/1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

