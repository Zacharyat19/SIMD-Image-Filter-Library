#pragma once

#include <string>
#include <algorithm>
#include <smmintrin.h>

#include <opencv2/opencv.hpp>

class Image
{
    public:
        Image(const std::string& imagePath);
        cv::Mat scalarBlur();
        cv::Mat scalarSharpen();
        cv::Mat scalarEdgeDetection();
        cv::Mat simdBlur();
        cv::Mat simdSharpen();
        cv::Mat simdEdgeDetection();
        cv::Mat getImage()
        {
            return image;
        }
    private:
        cv::Mat image;
};