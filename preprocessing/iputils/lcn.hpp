#pragma once
#ifndef __IPUTILS_LCN__
#define __IPUTILS_LCN__

#include <opencv2/opencv.hpp>

//! implements Local Contrast Normalization technique
// Description: http://bigwww.epfl.ch/sage/soft/localnormalization/
/*!
\param k_size divide image size by this value to operate faster.
\return 
*/
cv::Mat normalize_local_contrast(cv::Mat const& image, int const k_size = 13);
cv::Mat normalize_global_contrast(cv::Mat const& image);
cv::Mat scale_min_max(cv::Mat const& img);

extern const double gaussian_99_percentile;

#endif

