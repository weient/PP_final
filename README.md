# Image Filters Parallel Acceleration

This project implements 6 different image filters with parallel acceleration using CUDA and OpenMP.

## Project Demo

[![Image Filters Acceleration Demo](https://github.com/weient/PP_final/blob/main/demo.png)](https://youtu.be/2lXhEFKX0TE)

Click the image above to watch the demonstration video.

## Implemented Filters

We implemented 6 different image filters with parallel acceleration:

1. **Gaussian Blur** - Applies a smoothing effect using a Gaussian function
   - Parameters: Radius, Sigma

2. **Emboss** - Creates a 3D-like effect highlighting edges with raised or indented look
   - Parameters: Intensity

3. **Erosion** - Morphological operation that erodes away boundaries of foreground objects
   - Parameters: Radius

4. **Dilation** - Morphological operation that expands the boundaries of foreground objects
   - Parameters: Radius

5. **Wave** - Creates a wavy distortion effect
   - Parameters: Frequency, Amplitude

6. **Oil Painting** - Simulates an oil painting artistic effect
   - Parameters: Radius

## Implementation Details

The filters are categorized into three types of operations:

### Min/Max Operations
- **Erosion**: Finds the minimum value in a kernel neighborhood
- **Dilation**: Finds the maximum value in a kernel neighborhood

### Weighted Combination
- **Gaussian Blur**: Uses a Gaussian kernel for weighted averaging
- **Emboss**: Uses a specific kernel to create a 3D effect

### Special Operations
- **Wave**: Uses pixel displacement with sine functions
- **Oil Painting**: Uses intensity-based operation with histogram analysis

## Optimization Techniques

### CUDA Optimizations
- **Shared Memory**: Reduced global memory access by loading blocks into shared memory
- **Thread Block Optimization**: Configured optimal thread block dimensions (32×32)
- **Loop Unrolling**: Reduced loop overhead and improved instruction-level parallelism
- **Coalesced Memory Access**: Ensured aligned memory access patterns
- **Blocking Factor Tuning**: Experimented with different blocking factors to find the optimal size

### OpenMP Optimizations
- Parallel processing using multiple CPU cores
- Dynamic scheduling for load balancing

## Performance Results

### OpenMP Performance
- Near-linear speedup with increasing CPU cores
- 12.07× speedup with 12 CPU cores compared to sequential implementation

### CUDA Performance
- Baseline CUDA implementation: 14.54× speedup over sequential CPU
- Optimized CUDA implementation: 37.00× speedup over sequential CPU
- Key improvements:
  - Shared memory: 33.26× speedup
  - Block size optimization: 34.97× speedup
  - Loop unrolling: 35.05× speedup
  - Coalesced memory: 37.00× speedup

### Time Distribution
- Optimized CUDA implementation significantly reduced both computing time and I/O overhead
- Computing time reduced from 2.44s to 0.97s
- I/O time reduced from 2.80s to 2.78s

## Technical Implementation Details

### Setup
```cpp
// Thread block & grid dimension
dim3 block(32, 32);
dim3 grid(
    (width + 32 - 1) / 32,
    (height + 32 - 1) / 32
);

// Calculate shared memory tile dimensions
const int tile_w = BLOCK_DIM_X + 2 * radius;
const int tile_h = BLOCK_DIM_Y + 2 * radius;
```

### Memory Management
- Each thread block loads a tile of (32 + 2 × radius) × (32 + 2 × radius) pixels into shared memory
- Each thread processes one output pixel using this shared memory data

### Filter Operations Examples
- **Gaussian Blur**:
  ```cpp
  // Compute filter on CPU
  for(int y = -radius; y <= radius; y++) {
      for(int x = -radius; x <= radius; x++) {
          float value = exp(-(x*x + y*y)/(2*sigma*sigma));
          kernel[(y+radius) * kernelSize + (x+radius)] = value;
          sum += value;
      }
  }
  ```

- **Wave Effect**:
  ```cpp
  // Kernel code
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sourceX = x + amplitudeX * sin(y * frequencyY);
  int sourceY = y + amplitudeY * sin(x * frequencyX);
  Output[y][x] = Input[sourceY][sourceX];
  ```

## Development Environment

- CUDA Toolkit
- GCC with OpenMP support
- NVIDIA GPU (GTX 1080 used for benchmarking)

## Team Information

**Team 8**
- 112062520 戴維恩
- 111062698 戴樂為
