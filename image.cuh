#ifndef IMAGE_H
#define IMAGE_H

#include <png.h>

typedef struct {
    unsigned char* data;
    unsigned width;
    unsigned height;
    unsigned channels;
} Image;

void freeImage(Image* img);
Image* createImage(unsigned width, unsigned height, unsigned channels);
int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels);
void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels);

#endif