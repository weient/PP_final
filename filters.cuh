#ifndef CUDA_FILTERS_CUH
#define CUDA_FILTERS_CUH
#include "image.cuh"

void gaussianBlurCUDA(Image* src, Image* dst, int radius, float sigma);
void gaussianBlurCUDAoptimize(Image* src, Image* dst, int radius, float sigma);

void embossCUDAoptimize(Image* src, Image* dst, float intensity);
void embossCUDA(Image* src, Image* dst, float intensity);

void erosionCUDAoptimize(Image* src, Image* dst, int radius);
void erosionCUDA(Image* src, Image* dst, int radius);

void dilationCUDAoptimize(Image* src, Image* dst, int radius);
void dilationCUDA(Image* src, Image* dst, int radius);

void waveCUDA(Image* src, Image* dst, float amplitudeX, float amplitudeY, float frequencyX, float frequencyY);

void oilPaintingCUDAoptimize(Image* src, Image* dst, int radius, int intensityLevels);
void oilPaintingCUDA(Image* src, Image* dst, int radius, int intensityLevels);

#endif