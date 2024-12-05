#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <string.h>
#include <time.h>


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

    // // Cache the kernel in shared memory
    // extern __shared__ float sharedKernel[];
    
    // // Each thread helps load a portion of the kernel
    // int kernelSize = (2 * radius + 1) * (2 * radius + 1);
    // for(int i = threadIdx.x; i < kernelSize; i += blockDim.x) {
    //     if(i < kernelSize) {
    //         sharedKernel[i] = kernel[i];
    //     }
    // }
    // __syncthreads();

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
            // sum += src[src_idx] * sharedKernel[kernel_idx];
        }
    }
    
    dst[idx] = (unsigned char)(sum + 0.5f);
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

    // Cache the 3x3 kernel in shared memory
    __shared__ float sharedKernel[3][3];
    
    // Only first thread in block initializes the kernel
    if (threadIdx.x == 0) {
        // Initialize kernel values
        sharedKernel[0][0] = -2.0f * intensity;
        sharedKernel[0][1] = -1.0f * intensity;
        sharedKernel[0][2] = 0.0f;
        sharedKernel[1][0] = -1.0f * intensity;
        sharedKernel[1][1] = 1.0f;
        sharedKernel[1][2] = intensity;
        sharedKernel[2][0] = 0.0f;
        sharedKernel[2][1] = intensity;
        sharedKernel[2][2] = 2.0f * intensity;
    }
    __syncthreads();
    
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
            // sum += src[src_idx] * kernel[ky+1][kx+1];
            sum += src[src_idx] * sharedKernel[ky+1][kx+1];
        }
    }
    
    sum = min(max(sum, 0.0f), 255.0f);
    dst[idx] = (unsigned char)sum;
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
                
    // Find maximum value in the neighborhood
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
    if(sourceX < 0) sourceX = 0;
    if(sourceX >= width) sourceX = width - 1;
    if(sourceY < 0) sourceY = 0;
    if(sourceY >= height) sourceY = height - 1;
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
        gaussianBlurCUDA(src, dst, radius, sigma);  // Changed from gaussianBlur
    }
    // ./filter input.png output.png emboss 1.0
    else if(strcmp(filter_type, "emboss") == 0 && argc == 5) {
        float intensity = atof(argv[4]);
        embossCUDA(src, dst, intensity);  // Changed from emboss
    }
    // ./filter input.png output.png erode 2
    else if(strcmp(filter_type, "erode") == 0 && argc == 5) {
        int radius = atoi(argv[4]);
        erosionCUDA(src, dst, radius);  // Changed from erosion
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

