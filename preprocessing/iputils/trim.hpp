#pragma once
#ifndef __IPUTILS_TRIM__
#define __IPUTILS_TRIM__

#include <opencv2/opencv.hpp>

//! Trim constant color image borders.
/*!
\param divider divide image size by this value to operate faster.
\return image with borders removed
*/
cv::Mat trim(cv::Mat const& image, unsigned const divider = 8);

#endif

