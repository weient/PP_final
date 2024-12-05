#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH
#include "image.cuh"

__global__ void gaussianKernelOptimized(unsigned char* src, unsigned char* dst, int width, int height, int channels, int radius, float* kernel);
__global__ void gaussianKernel(unsigned char* src, unsigned char* dst, int width, int height, int channels, int radius, float sigma, float* kernel);

__global__ void embossKernelOptimized(unsigned char* src, unsigned char* dst,int width, int height, int channels, float intensity);
__global__ void embossKernel(unsigned char* src, unsigned char* dst,int width, int height, int channels, float intensity);

__global__ void erosionKernelOptimized(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius);
__global__ void erosionKernel(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius);

__global__ void dilationKernelOptimized(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius);
__global__ void dilationKernel(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius);

__global__ void waveKernel(unsigned char* src, unsigned char* dst,int width, int height, int channels, float amplitudeX, float amplitudeY, float frequencyX, float frequencyY);

__global__ void oilPaintingKernelOptimized(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius, int intensityLevels);
__global__ void oilPaintingKernel(unsigned char* src, unsigned char* dst,int width, int height, int channels, int radius, int intensityLevels);

#endif