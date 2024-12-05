#include <cuda_runtime.h>
#include "kernels.cuh"
#include "image.cuh"
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define CHANNELS 3

__global__ void gaussianKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius,
    float sigma,
    float* kernel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;
    
    if (idx >= total_pixels) return;

    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;
    
    float sum = 0.0f;
    
    for(int ky = -radius; ky <= radius; ky++) {
        for(int kx = -radius; kx <= radius; kx++) {
            int py = min(max(y + ky, 0), height - 1);
            int px = min(max(x + kx, 0), width - 1);
            
            int kernel_idx = (ky + radius) * (2 * radius + 1) + (kx + radius);
            int src_idx = (py * width + px) * channels + c;
            
            sum += src[src_idx] * kernel[kernel_idx];
        }
    }
    
    dst[idx] = (unsigned char)(sum + 0.5f);
}

__global__ void gaussianKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius,
    float* kernel
) {
    extern __shared__ unsigned char shared_mem[];
    
    // Calculate tile dimensions
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    const int kernelSize = 2 * radius + 1;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    // Calculate shared memory layout
    const int kernel_size = kernelSize * kernelSize;
    float* shared_kernel = (float*)shared_mem;
    unsigned char* shared_image = (unsigned char*)&shared_kernel[kernel_size];
    
    // Cooperatively load kernel into shared memory
    for(int i = threadIdx.y * BLOCK_DIM_X + threadIdx.x; 
        i < kernel_size; 
        i += BLOCK_DIM_X * BLOCK_DIM_Y) {
        if(i < kernel_size) {
            shared_kernel[i] = kernel[i];
        }
    }
    
    __syncthreads();
    
    // Load image data into shared memory
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &shared_image[c * tile_w * tile_h];
        
        for (int dy = ty; dy < tile_h; dy += BLOCK_DIM_Y) {
            for (int dx = tx; dx < tile_w; dx += BLOCK_DIM_X) {
                int gy = by + dy - radius;
                int gx = bx + dx - radius;
                
                // Clamp coordinates to image boundaries
                gy = max(0, min(gy, height - 1));
                gx = max(0, min(gx, width - 1));
                
                smem_c[dy * tile_w + dx] = src[(gy * width + gx) * channels + c];
            }
        }
    }
    
    __syncthreads();
    
    // Check if this thread processes a valid pixel
    if (x >= width || y >= height) return;
    
    // Process each channel
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &shared_image[c * tile_w * tile_h];
        float sum = 0.0f;
        int ki = 0;
        
        // Apply Gaussian filter using shared memory
        #pragma unroll
        for (int ky = 0; ky < kernelSize; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernelSize; kx++) {
                const int sy = ty + ky;
                const int sx = tx + kx;
                const float k_val = shared_kernel[ki++];
                
                float pixel_value = (float)smem_c[sy * tile_w + sx];
                sum += pixel_value * k_val;
            }
        }
        
        // Write result with proper rounding
        dst[(y * width + x) * channels + c] = (unsigned char)(min(max(sum + 0.5f, 0.0f), 255.0f));
    }
}

__global__ void embossKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    float intensity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;

    if (idx >= total_pixels) return;
    
    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;
    
    // Skip border pixels
    if(x == 0 || x == width-1 || y == 0 || y == height-1) {
        dst[idx] = src[idx];
        return;
    }
    
    float side = -2.0f * intensity;
    float corner = -1.0f * intensity;
    float center = 1.0f;
    float opposite = 2.0f * intensity;
    
    float kernel[3][3] = {
        {side, corner, 0},
        {corner, center, intensity},
        {0, intensity, opposite}
    };
    
    float sum = 128.0f; // Add bias to avoid negative values
    
    for(int ky = -1; ky <= 1; ky++) {
        for(int kx = -1; kx <= 1; kx++) {
            int src_idx = ((y + ky) * width + (x + kx)) * channels + c;
            sum += src[src_idx] * kernel[ky+1][kx+1];
        }
    }
    
    sum = min(max(sum, 0.0f), 255.0f);
    dst[idx] = (unsigned char)sum;
}

__global__ void embossKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    float intensity
) {
    extern __shared__ unsigned char smem[];
    
    // Define tile dimensions (similar to erosion filter)
    const int radius = 1; // Emboss uses 3x3 filter, so radius is 1
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    // Load filter coefficients into shared memory
    __shared__ float filter[3][3];
    if (tx == 0 && ty == 0) {
        float side = -2.0f * intensity;
        float corner = -1.0f * intensity;
        float center = 1.0f;
        float opposite = 2.0f * intensity;
        
        filter[0][0] = side;
        filter[0][1] = corner;
        filter[0][2] = 0.0f;
        filter[1][0] = corner;
        filter[1][1] = center;
        filter[1][2] = intensity;
        filter[2][0] = 0.0f;
        filter[2][1] = intensity;
        filter[2][2] = opposite;
    }
    __syncthreads();
    
    // Load image data into shared memory
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        // Collaborative loading of image tile into shared memory
        for (int dy = ty; dy < tile_h; dy += BLOCK_DIM_Y) {
            for (int dx = tx; dx < tile_w; dx += BLOCK_DIM_X) {
                int gy = by + dy - radius;
                int gx = bx + dx - radius;
                
                // Clamp coordinates to image boundaries
                gy = max(0, min(gy, height - 1));
                gx = max(0, min(gx, width - 1));
                
                smem_c[dy * tile_w + dx] = src[(gy * width + gx) * channels + c];
            }
        }
    }
    
    __syncthreads();
    
    // Check if this thread processes a valid pixel
    if (x >= width || y >= height) return;
    
    // Process each channel
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        float sum = 128.0f; // Add bias to avoid negative values
        
        // Apply filter using shared memory
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                const int sy = (ty + radius + ky);
                const int sx = (tx + radius + kx);
                float val = (float)smem_c[sy * tile_w + sx];
                sum += val * filter[ky + radius][kx + radius];
            }
        }
        
        // Clamp and write result
        sum = min(max(sum, 0.0f), 255.0f);
        dst[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

__global__ void erosionKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;
    
    if (idx >= total_pixels) return;
    
    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;
    
    unsigned char min_val = 255;
    
    for(int ky = -radius; ky <= radius; ky++) {
        for(int kx = -radius; kx <= radius; kx++) {
            int py = min(max(y + ky, 0), height - 1);
            int px = min(max(x + kx, 0), width - 1);
            
            unsigned char val = src[(py * width + px) * channels + c];
            min_val = min(val, min_val);
        }
    }
    
    dst[idx] = min_val;
}

__global__ void erosionKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius
) {
    extern __shared__ unsigned char smem[];
    
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        for (int dy = ty; dy < tile_h; dy += BLOCK_DIM_Y) {
            for (int dx = tx; dx < tile_w; dx += BLOCK_DIM_X) {
                int gy = by + dy - radius;
                int gx = bx + dx - radius;
                
                gy = max(0, min(gy, height - 1));
                gx = max(0, min(gx, width - 1));
                
                smem_c[dy * tile_w + dx] = src[(gy * width + gx) * channels + c];
            }
        }
    }
    
    __syncthreads();
    if (x >= width || y >= height) return;
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char min_val = 255;
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                const int sy = (ty + radius + ky);
                const int sx = (tx + radius + kx);
                min_val = min(min_val, smem_c[sy * tile_w + sx]);
            }
        }
        
        dst[(y * width + x) * channels + c] = min_val;
    }
}

__global__ void dilationKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius
) {
    extern __shared__ unsigned char smem[];
    
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        for (int dy = ty; dy < tile_h; dy += BLOCK_DIM_Y) {
            for (int dx = tx; dx < tile_w; dx += BLOCK_DIM_X) {
                int gy = by + dy - radius;
                int gx = bx + dx - radius;
                
                gy = max(0, min(gy, height - 1));
                gx = max(0, min(gx, width - 1));
                
                smem_c[dy * tile_w + dx] = src[(gy * width + gx) * channels + c];
            }
        }
    }
    
    __syncthreads();
    if (x >= width || y >= height) return;
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char max_val = 0;
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                const int sy = (ty + radius + ky);
                const int sx = (tx + radius + kx);
                max_val = max(max_val, smem_c[sy * tile_w + sx]);
            }
        }
        
        dst[(y * width + x) * channels + c] = max_val;
    }
}

__global__ void dilationKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;
    
    if (idx >= total_pixels) return;
    
    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;

    unsigned char max_val = 0;
                
    for(int ky = -radius; ky <= radius; ky++) {
        for(int kx = -radius; kx <= radius; kx++) {
            int py = min(max(y + ky, 0), height - 1);
            int px = min(max(x + kx, 0), width - 1);
            
            unsigned char val = src[(py * width + px) * channels + c];
            max_val = max(val, max_val);
        }
    }
    
    dst[(y * width + x) * channels + c] = max_val;
}

__global__ void waveKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    float amplitudeX, 
    float amplitudeY, 
    float frequencyX, 
    float frequencyY
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;
    if (idx >= total_pixels) return;
    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;

    int sourceX = x + amplitudeX * sin(y * frequencyY);
    int sourceY = y + amplitudeY * sin(x * frequencyX);
    
    sourceX = max(min(sourceX, width - 1), 0);
    sourceY = max(min(sourceY, height - 1), 0);

    dst[(y * width + x) * channels + c] = 
        src[(sourceY * width + sourceX) * channels + c];

}

__global__ void oilPaintingKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius,
    int intensityLevels
) {

    extern __shared__ unsigned char smem[];
    
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    #pragma unroll
    for (int c = 0; c < channels; c++) {

        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        for (int dy = ty; dy < tile_h; dy += BLOCK_DIM_Y) {
            for (int dx = tx; dx < tile_w; dx += BLOCK_DIM_X) {
                int gy = by + dy - radius;
                int gx = bx + dx - radius;
                
                gy = max(0, min(gy, height - 1));
                gx = max(0, min(gx, width - 1));
                
                smem_c[dy * tile_w + dx] = src[(gy * width + gx) * channels + c];
            }
        }
    }
    
    __syncthreads();
    if (x >= width || y >= height) return;

    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        int intensity_count[256] = {0};
        unsigned long long intensity_sum[256] = {0};
        
        for(int ky = -radius; ky <= radius; ky++) {
            for(int kx = -radius; kx <= radius; kx++) {
                const int sy = (ty + radius + ky);
                const int sx = (tx + radius + kx);
                unsigned char val = smem_c[sy * tile_w + sx];
                int intensity = (val * intensityLevels) / 256;
                intensity_count[intensity]++;
                intensity_sum[intensity] += val;
            }
        }
        
        int max_count = 0;
        int max_intensity = 0;
        for(int i = 0; i < intensityLevels; i++) {
            if(intensity_count[i] > max_count) {
                max_count = intensity_count[i];
                max_intensity = i;
            }
        }
        int idx = (y * width + x) * channels + c;
        dst[idx] = max_count ? 
            (unsigned char)(intensity_sum[max_intensity] / intensity_count[max_intensity]) : 
            src[idx];
    }
}

__global__ void oilPaintingKernel(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius,
    int intensityLevels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * channels;
    
    if (idx >= total_pixels) return;
    
    int c = idx % channels;
    int xy = idx / channels;
    int x = xy % width;
    int y = xy / width;
    
    int intensity_count[256] = {0};
    unsigned long long intensity_sum[256] = {0};
    
    for(int ky = -radius; ky <= radius; ky++) {
        for(int kx = -radius; kx <= radius; kx++) {
            int py = min(max(y + ky, 0), height - 1);
            int px = min(max(x + kx, 0), width - 1);
            
            int nidx = (py * width + px) * channels + c;
            int intensity = (src[nidx] * intensityLevels) / 256;
            
            intensity_count[intensity]++;
            intensity_sum[intensity] += src[nidx];
        }
    }
    
    int max_count = 0;
    int max_intensity = 0;
    for(int i = 0; i < intensityLevels; i++) {
        if(intensity_count[i] > max_count) {
            max_count = intensity_count[i];
            max_intensity = i;
        }
    }
    
    dst[idx] = max_count ? 
        (unsigned char)(intensity_sum[max_intensity] / intensity_count[max_intensity]) : 
        src[idx];
}

