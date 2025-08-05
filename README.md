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
