#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <string.h>
#include <time.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define CHANNELS 3

typedef struct {
    unsigned char* data;
    unsigned width;
    unsigned height;
    unsigned channels;
} Image;

Image* createImage(unsigned width, unsigned height, unsigned channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (unsigned char*)malloc(width * height * channels);
    return img;
}

void freeImage(Image* img) {
    free(img->data);
    free(img);
}

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {
    FILE* infile = fopen(filename, "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    
    png_init_io(png_ptr, infile);
    png_read_info(png_ptr, info_ptr);

    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);
    
    // Update info after transformations
    png_read_update_info(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);
    png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    
    *image = (unsigned char*)malloc(rowbytes * *height);
    png_bytepp row_pointers = (png_bytepp)malloc(*height * sizeof(png_bytep));
    
    for (unsigned int i = 0; i < *height; i++) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(infile);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    
    png_init_io(png_ptr, fp);
    
    int color_type = channels == 4 ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type, 
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (unsigned int i = 0; i < height; i++) {
        row_pointers[i] = image + i * width * channels;
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void erosion(Image* src, Image* dst, int radius) {
    for(int y = 0; y < src->height; y++) {
        for(int x = 0; x < src->width; x++) {
            for(int c = 0; c < src->channels; c++) {
                unsigned char min_val = 255;
                
                // Find minimum value in the neighborhood
                for(int ky = -radius; ky <= radius; ky++) {
                    for(int kx = -radius; kx <= radius; kx++) {
                        int py = y + ky;
                        int px = x + kx;
                        
                        // Border handling
                        if(px < 0) px = 0;
                        if(px >= src->width) px = src->width - 1;
                        if(py < 0) py = 0;
                        if(py >= src->height) py = src->height - 1;
                        
                        unsigned char val = src->data[(py * src->width + px) * src->channels + c];
                        if(val < min_val) {
                            min_val = val;
                        }
                    }
                }
                
                dst->data[(y * dst->width + x) * dst->channels + c] = min_val;
            }
        }
    }
}
void gaussianBlur(Image* src, Image* dst, int radius, float sigma) {
    int kernelSize = 2 * radius + 1;
    float** kernel = (float**)malloc(kernelSize * sizeof(float*));
    for(int i = 0; i < kernelSize; i++) {
        kernel[i] = (float*)malloc(kernelSize * sizeof(float));
    }
    
    // Calculate kernel
    float sum = 0.0;
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            float value = exp(-(x*x + y*y)/(2*sigma*sigma));
            kernel[y+radius][x+radius] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for(int y = 0; y < kernelSize; y++) {
        for(int x = 0; x < kernelSize; x++) {
            kernel[y][x] /= sum;
        }
    }
    
    // Apply blur
    for(int y = 0; y < src->height; y++) {
        for(int x = 0; x < src->width; x++) {
            for(int c = 0; c < src->channels; c++) {
                float sum = 0.0;
                
                for(int ky = -radius; ky <= radius; ky++) {
                    for(int kx = -radius; kx <= radius; kx++) {
                        int py = y + ky;
                        int px = x + kx;
                        
                        if(px < 0) px = 0;
                        if(px >= src->width) px = src->width - 1;
                        if(py < 0) py = 0;
                        if(py >= src->height) py = src->height - 1;
                        
                        int srcIdx = (py * src->width + px) * src->channels + c;
                        sum += src->data[srcIdx] * kernel[ky+radius][kx+radius];
                    }
                }
                
                int dstIdx = (y * dst->width + x) * dst->channels + c;
                dst->data[dstIdx] = (unsigned char)(sum + 0.5f);
            }
        }
    }
    
    for(int i = 0; i < kernelSize; i++) {
        free(kernel[i]);
    }
    free(kernel);
}
void emboss(Image* src, Image* dst) {
    float kernel[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
    
    for(int y = 1; y < src->height-1; y++) {
        for(int x = 1; x < src->width-1; x++) {
            for(int c = 0; c < src->channels; c++) {
                float sum = 128; // Add bias to avoid negative values
                
                for(int ky = -1; ky <= 1; ky++) {
                    for(int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * src->width + (x + kx)) * src->channels + c;
                        sum += src->data[idx] * kernel[ky+1][kx+1];
                    }
                }
                
                if(sum < 0) sum = 0;
                if(sum > 255) sum = 255;
                dst->data[(y * dst->width + x) * dst->channels + c] = (unsigned char)sum;
            }
        }
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
    // Dynamically sized shared memory - declared externally
    extern __shared__ unsigned char smem[];
    
    // Calculate padded tile dimensions
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    // Calculate global and local indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    // Load data into shared memory including halo region
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        // Calculate base offset for each channel in shared memory
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        // Each thread loads multiple pixels to cover tile + halo
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
    if (x >= width || y >= height) return;
    // Process only valid pixels
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        unsigned char max_val = 0;
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        // Use shared memory for neighborhood search
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

void dilationCUDA(Image* src, Image* dst, int radius) {
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
    
    // Bounds checking
    sourceX = max(min(sourceX, width - 1), 0);
    sourceY = max(min(sourceY, height - 1), 0);

    dst[(y * width + x) * channels + c] = 
        src[(sourceY * width + sourceX) * channels + c];

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

__global__ void oilPaintingKernelOptimized(
    unsigned char* src,
    unsigned char* dst,
    int width,
    int height,
    int channels,
    int radius,
    int intensityLevels
) {
    // Dynamically sized shared memory - declared externally
    extern __shared__ unsigned char smem[];
    
    // Calculate padded tile dimensions
    const int tile_w = BLOCK_DIM_X + 2 * radius;
    const int tile_h = BLOCK_DIM_Y + 2 * radius;
    
    // Calculate global and local indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_DIM_X;
    const int by = blockIdx.y * BLOCK_DIM_Y;
    const int x = bx + tx;
    const int y = by + ty;
    
    // Load data into shared memory including halo region
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        // Calculate base offset for each channel in shared memory
        unsigned char* smem_c = &smem[c * tile_w * tile_h];
        
        // Each thread loads multiple pixels to cover tile + halo
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
    if (x >= width || y >= height) return;
    // Process only valid pixels

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

int main(int argc, char* argv[]) {
    if(argc != 4 && argc != 5 && argc != 6 && argc != 7) {
        printf("input error\n");
        return 1;
    }

    unsigned char* buffer;
    unsigned width, height, channels;
    
    read_png(argv[1], &buffer, &height, &width, &channels);
    
    Image* src = createImage(width, height, channels);
    Image* dst = createImage(width, height, channels);
    
    memcpy(src->data, buffer, width * height * channels);
    memcpy(dst->data, buffer, width * height * channels); // Initialize dst with source
    free(buffer);
    
    const char* filter_type = argv[3];
    
    // ./filter input.png output.png gaussian 5 3.0
    if(strcmp(filter_type, "gaussian") == 0 && argc == 6) {
        int radius = atoi(argv[4]);
        float sigma = atof(argv[5]);
        gaussianBlur(src, dst, radius, sigma);
    }
    // ./filter input.png output.png emboss
    else if(strcmp(filter_type, "emboss") == 0) {
        emboss(src, dst);
    }
    // ./filter input.png output.png erode 2
    else if(strcmp(filter_type, "erode") == 0 && argc == 5) {
        int radius = atoi(argv[4]);
        erosion(src, dst, radius);
    }
    // ./filter input.png output.png dilate 2
    else if(strcmp(filter_type, "dilate") == 0 && argc == 5) {
        int radius = atoi(argv[4]);
        dilationCUDA(src, dst, radius);
    }
    // ./filter input.png output.png oil 3
    else if(strcmp(filter_type, "oil") == 0 && argc == 5) {
        int radius = atoi(argv[4]);
        oilPaintingCUDA(src, dst, radius, 20);
    }

    // ./filter input.png output.png wave 10 10 0.1
    else if(strcmp(filter_type, "wave") == 0 && argc == 7) {
        float ampX = atof(argv[4]);
        float ampY = atof(argv[5]);
        float freqX = atof(argv[6]);
        waveCUDA(src, dst, ampX, ampY, freqX, freqX);
    }
    else {
        printf("Invalid filter type or parameters\n");
        return 1;
    }
    
    write_png(argv[2], dst->data, height, width, channels);
    
    freeImage(src);
    freeImage(dst);
    return 0;
}

