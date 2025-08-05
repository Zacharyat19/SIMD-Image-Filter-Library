#include "../Inc/Image.h"

Image::Image(const std::string& imagePath)
{
    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) 
    {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
}

cv::Mat Image::scalarBlur()
{
    cv::Mat output = image.clone();

    for (int y = 1; y < image.rows - 1; ++y) 
    {
        for (int x = 1; x < image.cols - 1; ++x) 
        {
            int sum = 0;

            for (int ky = -1; ky <= 1; ++ky) 
            {
                for (int kx = -1; kx <= 1; ++kx) 
                {
                    sum += image.at<uchar>(y + ky, x + kx);
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum / 9);
        }
    }

    return output;
}

cv::Mat Image::scalarSharpen() 
{
    cv::Mat output = image.clone();

    for (int y = 1; y < image.rows - 1; ++y) 
    {
        for (int x = 1; x < image.cols - 1; ++x) 
        {
            int center = 5 * image.at<uchar>(y, x);
            int north  = image.at<uchar>(y - 1, x);
            int south  = image.at<uchar>(y + 1, x);
            int west   = image.at<uchar>(y, x - 1);
            int east   = image.at<uchar>(y, x + 1);

            int value = center - north - south - east - west;

            value = std::min(255, std::max(0, value));

            output.at<uchar>(y, x) = static_cast<uchar>(value);
        }
    }

    return output;
}

cv::Mat Image::scalarEdgeDetection() 
{
    cv::Mat output = cv::Mat::zeros(image.size(), image.type());

    for (int y = 1; y < image.rows - 1; ++y) 
    {
        for (int x = 1; x < image.cols - 1; ++x) 
        {
            int gx = 0, gy = 0;

            // Gx
            gx += -1 * image.at<uchar>(y - 1, x - 1);
            gx +=  0 * image.at<uchar>(y - 1, x);
            gx +=  1 * image.at<uchar>(y - 1, x + 1);

            gx += -2 * image.at<uchar>(y, x - 1);
            gx +=  0 * image.at<uchar>(y, x);
            gx +=  2 * image.at<uchar>(y, x + 1);

            gx += -1 * image.at<uchar>(y + 1, x - 1);
            gx +=  0 * image.at<uchar>(y + 1, x);
            gx +=  1 * image.at<uchar>(y + 1, x + 1);

            // Gy
            gy += -1 * image.at<uchar>(y - 1, x - 1);
            gy += -2 * image.at<uchar>(y - 1, x);
            gy += -1 * image.at<uchar>(y - 1, x + 1);

            gy +=  0 * image.at<uchar>(y, x - 1);
            gy +=  0 * image.at<uchar>(y, x);
            gy +=  0 * image.at<uchar>(y, x + 1);

            gy +=  1 * image.at<uchar>(y + 1, x - 1);
            gy +=  2 * image.at<uchar>(y + 1, x);
            gy +=  1 * image.at<uchar>(y + 1, x + 1);

            int mag = std::abs(gx) + std::abs(gy);

            mag = std::min(255, std::max(0, mag));

            output.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }

    return output;
}

cv::Mat Image::simdBlur() 
{
    cv::Mat output = image.clone();

    const int rows = image.rows;
    const int cols = image.cols;

    for (int y = 1; y < rows - 1; ++y) 
    {
        int x = 1;

        for (; x <= cols - 1 - 16; x += 16) 
        {
            __m128i sum = _mm_setzero_si128();

            for (int ky = -1; ky <= 1; ++ky) 
            {
                for (int kx = -1; kx <= 1; ++kx) 
                {
                    const uchar* srcPtr = image.ptr<uchar>(y + ky) + x + kx;
                    __m128i pixels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcPtr));
                    sum = _mm_add_epi16(sum, _mm_unpacklo_epi8(pixels, _mm_setzero_si128()));
                }
            }

            alignas(16) uint16_t temp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(temp), sum);

            uchar result[16];
            for (int i = 0; i < 16; ++i) 
            {
                uint16_t val = ((uint16_t*)&sum)[i];
                result[i] = static_cast<uchar>(val / 9);
            }

            uchar* dstPtr = output.ptr<uchar>(y) + x;
            memcpy(dstPtr, result, 16);
        }

        for (; x < cols - 1; ++x) 
        {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    sum += image.at<uchar>(y + ky, x + kx);
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum / 9);
        }
    }

    return output;
}

cv::Mat Image::simdSharpen() 
{
    cv::Mat output = image.clone();

    const int rows = image.rows;
    const int cols = image.cols;

    for (int y = 1; y < rows - 1; ++y) 
    {
        int x = 1;

        for (; x <= cols - 1 - 16; x += 16) 
        {
            const uchar* centerPtr = image.ptr<uchar>(y) + x;
            __m128i center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(centerPtr));

            __m128i north = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y - 1) + x));
            __m128i south = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y + 1) + x));

            __m128i west = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y) + x - 1));
            __m128i east = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y) + x + 1));

            __m128i centerLo = _mm_unpacklo_epi8(center, _mm_setzero_si128());
            __m128i centerHi = _mm_unpackhi_epi8(center, _mm_setzero_si128());

            __m128i northLo = _mm_unpacklo_epi8(north, _mm_setzero_si128());
            __m128i northHi = _mm_unpackhi_epi8(north, _mm_setzero_si128());

            __m128i southLo = _mm_unpacklo_epi8(south, _mm_setzero_si128());
            __m128i southHi = _mm_unpackhi_epi8(south, _mm_setzero_si128());

            __m128i westLo = _mm_unpacklo_epi8(west, _mm_setzero_si128());
            __m128i westHi = _mm_unpackhi_epi8(west, _mm_setzero_si128());

            __m128i eastLo = _mm_unpacklo_epi8(east, _mm_setzero_si128());
            __m128i eastHi = _mm_unpackhi_epi8(east, _mm_setzero_si128());

            __m128i valLo = _mm_mullo_epi16(centerLo, _mm_set1_epi16(5));
            valLo = _mm_sub_epi16(valLo, northLo);
            valLo = _mm_sub_epi16(valLo, southLo);
            valLo = _mm_sub_epi16(valLo, westLo);
            valLo = _mm_sub_epi16(valLo, eastLo);

            __m128i valHi = _mm_mullo_epi16(centerHi, _mm_set1_epi16(5));
            valHi = _mm_sub_epi16(valHi, northHi);
            valHi = _mm_sub_epi16(valHi, southHi);
            valHi = _mm_sub_epi16(valHi, westHi);
            valHi = _mm_sub_epi16(valHi, eastHi);

            valLo = _mm_min_epi16(valLo, _mm_set1_epi16(255));
            valLo = _mm_max_epi16(valLo, _mm_setzero_si128());

            valHi = _mm_min_epi16(valHi, _mm_set1_epi16(255));
            valHi = _mm_max_epi16(valHi, _mm_setzero_si128());

            __m128i result = _mm_packus_epi16(valLo, valHi);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(output.ptr<uchar>(y) + x), result);
        }

        for (; x < cols - 1; ++x) 
        {
            int center = 5 * image.at<uchar>(y, x);
            int north = image.at<uchar>(y - 1, x);
            int south = image.at<uchar>(y + 1, x);
            int west = image.at<uchar>(y, x - 1);
            int east = image.at<uchar>(y, x + 1);

            int val = center - north - south - west - east;
            val = std::min(255, std::max(0, val));

            output.at<uchar>(y, x) = static_cast<uchar>(val);
        }
    }

    return output;
}

cv::Mat Image::simdEdgeDetection() 
{
    cv::Mat output = cv::Mat::zeros(image.size(), image.type());

    const int rows = image.rows;
    const int cols = image.cols;

    for (int y = 1; y < rows - 1; ++y) 
    {
        int x = 1;

        for (; x <= cols - 1 - 16; x += 16) 
        {
            __m128i p00 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y - 1) + x - 1));
            __m128i p01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y - 1) + x));
            __m128i p02 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y - 1) + x + 1));

            __m128i p10 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y) + x - 1));
            __m128i p11 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y) + x));
            __m128i p12 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y) + x + 1));

            __m128i p20 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y + 1) + x - 1));
            __m128i p21 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y + 1) + x));
            __m128i p22 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(image.ptr<uchar>(y + 1) + x + 1));

            __m128i p00Lo = _mm_unpacklo_epi8(p00, _mm_setzero_si128());
            __m128i p01Lo = _mm_unpacklo_epi8(p01, _mm_setzero_si128());
            __m128i p02Lo = _mm_unpacklo_epi8(p02, _mm_setzero_si128());
            __m128i p10Lo = _mm_unpacklo_epi8(p10, _mm_setzero_si128());
            __m128i p11Lo = _mm_unpacklo_epi8(p11, _mm_setzero_si128());
            __m128i p12Lo = _mm_unpacklo_epi8(p12, _mm_setzero_si128());
            __m128i p20Lo = _mm_unpacklo_epi8(p20, _mm_setzero_si128());
            __m128i p21Lo = _mm_unpacklo_epi8(p21, _mm_setzero_si128());
            __m128i p22Lo = _mm_unpacklo_epi8(p22, _mm_setzero_si128());

            __m128i p00Hi = _mm_unpackhi_epi8(p00, _mm_setzero_si128());
            __m128i p01Hi = _mm_unpackhi_epi8(p01, _mm_setzero_si128());
            __m128i p02Hi = _mm_unpackhi_epi8(p02, _mm_setzero_si128());
            __m128i p10Hi = _mm_unpackhi_epi8(p10, _mm_setzero_si128());
            __m128i p11Hi = _mm_unpackhi_epi8(p11, _mm_setzero_si128());
            __m128i p12Hi = _mm_unpackhi_epi8(p12, _mm_setzero_si128());
            __m128i p20Hi = _mm_unpackhi_epi8(p20, _mm_setzero_si128());
            __m128i p21Hi = _mm_unpackhi_epi8(p21, _mm_setzero_si128());
            __m128i p22Hi = _mm_unpackhi_epi8(p22, _mm_setzero_si128());

            __m128i gxLo = _mm_setzero_si128();
            gxLo = _mm_add_epi16(gxLo, _mm_mullo_epi16(p02Lo, _mm_set1_epi16(1)));
            gxLo = _mm_sub_epi16(gxLo, _mm_mullo_epi16(p00Lo, _mm_set1_epi16(1)));

            gxLo = _mm_add_epi16(gxLo, _mm_mullo_epi16(p12Lo, _mm_set1_epi16(2)));
            gxLo = _mm_sub_epi16(gxLo, _mm_mullo_epi16(p10Lo, _mm_set1_epi16(2)));

            gxLo = _mm_add_epi16(gxLo, _mm_mullo_epi16(p22Lo, _mm_set1_epi16(1)));
            gxLo = _mm_sub_epi16(gxLo, _mm_mullo_epi16(p20Lo, _mm_set1_epi16(1)));

            __m128i gxHi = _mm_setzero_si128();
            gxHi = _mm_add_epi16(gxHi, _mm_mullo_epi16(p02Hi, _mm_set1_epi16(1)));
            gxHi = _mm_sub_epi16(gxHi, _mm_mullo_epi16(p00Hi, _mm_set1_epi16(1)));

            gxHi = _mm_add_epi16(gxHi, _mm_mullo_epi16(p12Hi, _mm_set1_epi16(2)));
            gxHi = _mm_sub_epi16(gxHi, _mm_mullo_epi16(p10Hi, _mm_set1_epi16(2)));

            gxHi = _mm_add_epi16(gxHi, _mm_mullo_epi16(p22Hi, _mm_set1_epi16(1)));
            gxHi = _mm_sub_epi16(gxHi, _mm_mullo_epi16(p20Hi, _mm_set1_epi16(1)));

            __m128i gyLo = _mm_setzero_si128();
            gyLo = _mm_add_epi16(gyLo, _mm_mullo_epi16(p20Lo, _mm_set1_epi16(1)));
            gyLo = _mm_sub_epi16(gyLo, _mm_mullo_epi16(p00Lo, _mm_set1_epi16(1)));

            gyLo = _mm_add_epi16(gyLo, _mm_mullo_epi16(p21Lo, _mm_set1_epi16(2)));
            gyLo = _mm_sub_epi16(gyLo, _mm_mullo_epi16(p01Lo, _mm_set1_epi16(2)));

            gyLo = _mm_add_epi16(gyLo, _mm_mullo_epi16(p22Lo, _mm_set1_epi16(1)));
            gyLo = _mm_sub_epi16(gyLo, _mm_mullo_epi16(p02Lo, _mm_set1_epi16(1)));

            __m128i gyHi = _mm_setzero_si128();
            gyHi = _mm_add_epi16(gyHi, _mm_mullo_epi16(p20Hi, _mm_set1_epi16(1)));
            gyHi = _mm_sub_epi16(gyHi, _mm_mullo_epi16(p00Hi, _mm_set1_epi16(1)));

            gyHi = _mm_add_epi16(gyHi, _mm_mullo_epi16(p21Hi, _mm_set1_epi16(2)));
            gyHi = _mm_sub_epi16(gyHi, _mm_mullo_epi16(p01Hi, _mm_set1_epi16(2)));

            gyHi = _mm_add_epi16(gyHi, _mm_mullo_epi16(p22Hi, _mm_set1_epi16(1)));
            gyHi = _mm_sub_epi16(gyHi, _mm_mullo_epi16(p02Hi, _mm_set1_epi16(1)));

            gxLo = _mm_abs_epi16(gxLo);
            gxHi = _mm_abs_epi16(gxHi);
            gyLo = _mm_abs_epi16(gyLo);
            gyHi = _mm_abs_epi16(gyHi);

            __m128i magLo = _mm_add_epi16(gxLo, gyLo);
            __m128i magHi = _mm_add_epi16(gxHi, gyHi);

            magLo = _mm_min_epi16(magLo, _mm_set1_epi16(255));
            magHi = _mm_min_epi16(magHi, _mm_set1_epi16(255));

            __m128i result = _mm_packus_epi16(magLo, magHi);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(output.ptr<uchar>(y) + x), result);
        }

        for (; x < cols - 1; ++x) 
        {
            int gx = 0, gy = 0;

            gx += -1 * image.at<uchar>(y - 1, x - 1);
            gx += 0  * image.at<uchar>(y - 1, x);
            gx += 1  * image.at<uchar>(y - 1, x + 1);
            gx += -2 * image.at<uchar>(y, x - 1);
            gx += 0  * image.at<uchar>(y, x);
            gx += 2  * image.at<uchar>(y, x + 1);
            gx += -1 * image.at<uchar>(y + 1, x - 1);
            gx += 0  * image.at<uchar>(y + 1, x);
            gx += 1  * image.at<uchar>(y + 1, x + 1);

            gy += -1 * image.at<uchar>(y - 1, x - 1);
            gy += -2 * image.at<uchar>(y - 1, x);
            gy += -1 * image.at<uchar>(y - 1, x + 1);
            gy += 0  * image.at<uchar>(y, x - 1);
            gy += 0  * image.at<uchar>(y, x);
            gy += 0  * image.at<uchar>(y, x + 1);
            gy += 1  * image.at<uchar>(y + 1, x - 1);
            gy += 2  * image.at<uchar>(y + 1, x);
            gy += 1  * image.at<uchar>(y + 1, x + 1);

            int mag = std::abs(gx) + std::abs(gy);
            mag = std::min(255, std::max(0, mag));
            output.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }

    return output;
}