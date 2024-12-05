#include <stdlib.h>
#include "image.cuh"

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

void write_png(const char* filename, png_bytep image, const unsigned height, 
               const unsigned width, const unsigned channels) {
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