# SIMD Image Filters in C++

This project implements `blur`, `sharpen`, and `edge detection` image filters in C++ using both scalar (naive loop-based) and `SIMD` optimized versions. The filters operate on grayscale images using the OpenCV library for image loading and saving.

## Features

- Scalar and SIMD implementations of:
  - Blur (3×3 average)
  - Sharpen
  - Edge Detection (Sobel operator)
- CPU-timed performance metrics
- Output images written for easy comparison
- Uses **SSE4.1 intrinsics** for vectorization

## Build

Compile with`g++` using:
```bash
g++ -std=c++17 -O3 -msse4.1 src/main.cpp src/image.cpp -o simd_filters `pkg-config --cflags --libs opencv4`
```
## Performance Metrics

```bash
scalarBlur time: 50.9751 ms
scalarSharpen time: 18.089 ms
scalarEdgeDetection time: 24.2084 ms
simdBlur time: 6.7918 ms
simdSharpen time: 2.9173 ms
simdEdgeDetection time: 4.3886 ms
```
