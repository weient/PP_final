#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "image.cuh"
#include "filters.cuh"

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
    memcpy(dst->data, buffer, width * height * channels);
    free(buffer);
    
    const char* filter_type = argv[3];
    
    // ./filter_cuda input.png output.png gaussian 5 3.0
    if(strcmp(filter_type, "gaussian") == 0 && argc == 6) {
        gaussianBlurCUDAoptimize(src, dst, atoi(argv[4]), atof(argv[5]));
        // gaussianBlurCUDA(src, dst, atoi(argv[4]), atof(argv[5]));
    }
    // ./filter input.png output.png emboss 1.0
    else if(strcmp(filter_type, "emboss") == 0 && argc == 5) {
        embossCUDAoptimize(src, dst, atof(argv[4]));
        // embossCUDA(src, dst, atof(argv[4]));
    }
    // ./filter input.png output.png erode 2
    else if(strcmp(filter_type, "erode") == 0 && argc == 5) {
        erosionCUDAoptimize(src, dst, atoi(argv[4]));
        // erosionCUDA(src, dst, atoi(argv[4]));
    }
    // ./filter input.png output.png dilate 2
    else if(strcmp(filter_type, "dilate") == 0 && argc == 5) {
        dilationCUDA(src, dst, atoi(argv[4]));
        dilationCUDAoptimize(src, dst, atoi(argv[4]));
    }
    // ./filter input.png output.png oil 3
    else if(strcmp(filter_type, "oil") == 0 && argc == 5) {
        oilPaintingCUDA(src, dst, atoi(argv[4]), 20);
        oilPaintingCUDAoptimize(src, dst, atoi(argv[4]), 20);
    }
    // ./filter input.png output.png wave 10 10 0.1
    else if(strcmp(filter_type, "wave") == 0 && argc == 7) {
        waveCUDA(src, dst, atof(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[6]));
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