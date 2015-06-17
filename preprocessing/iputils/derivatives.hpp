#pragma once
#ifndef __IPUTILS_DERIVATIVES__
#define __IPUTILS_DERIVATIVES__

#include <opencv2/opencv.hpp>

typedef std::vector<cv::Mat> derivatives_t;
derivatives_t generate_derivatives(cv::Mat image);

//! Hessian blob detector
/*!
\param k_size gaussian kernel size
\return
*/
cv::Mat detect_blobes(cv::Mat const& image, double const k_size = 13, cv::Mat const& ignore_mask = cv::Mat(), derivatives_t& generate_derivatives = derivatives_t());

//! Hessian ridge detector
/*!
\param k_size gaussian kernel size
\return
*/
cv::Mat detect_ridges(cv::Mat const& image, double const k_size = 13, cv::Mat const& ignore_mask = cv::Mat(), derivatives_t& generate_derivatives = derivatives_t());

#endif

