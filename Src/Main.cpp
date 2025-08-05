#include <memory>
#include <iostream>
#include <chrono>

#include "../Inc/Image.h"

int main()
{
    std::unique_ptr<Image> filters = std::make_unique<Image>("Images/input.jpg");

    // Scalar blur
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat outputBlur = filters->scalarBlur();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/output_blur.jpg", outputBlur)) 
        {
            std::cerr << "Failed to save blurred image\n";
            return -1;
        }

        std::cout << "scalarBlur time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    // Scalar sharpen
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat outputSharpen = filters->scalarSharpen();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/output_sharp.jpg", outputSharpen)) 
        {
            std::cerr << "Failed to save sharpened image\n";
            return -1;
        }

        std::cout << "scalarSharpen time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    // Scalar edge detection
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat outputEdge = filters->scalarEdgeDetection();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/output_edge.jpg", outputEdge)) 
        {
            std::cerr << "Failed to save edge image\n";
            return -1;
        }

        std::cout << "scalarEdgeDetection time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    // SIMD blur
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat simdBlur = filters->simdBlur();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/simd_blur.jpg", simdBlur)) 
        {
            std::cerr << "Failed to save SIMD blurred image\n";
            return -1;
        }

        std::cout << "simdBlur time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    // SIMD sharpen
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat simdSharpen = filters->simdSharpen();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/simd_sharp.jpg", simdSharpen)) 
        {
            std::cerr << "Failed to save SIMD sharpened image\n";
            return -1;
        }

        std::cout << "simdSharpen time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    // SIMD edge detection
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat simdEdge = filters->simdEdgeDetection();
        auto end = std::chrono::high_resolution_clock::now();

        if (!cv::imwrite("Images/simd_edge.jpg", simdEdge)) 
        {
            std::cerr << "Failed to save SIMD edge image\n";
            return -1;
        }

        std::cout << "simdEdgeDetection time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    }

    return 0;
}