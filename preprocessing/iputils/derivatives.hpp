#pragma once
#ifndef __IPUTILS_DERIVATIVES__
#define __IPUTILS_DERIVATIVES__

#include <opencv2/opencv.hpp>

struct derivatives_t { cv::Mat dxx, dxy, dyy; double sigma; };
derivatives_t generate_derivatives(cv::Mat image);


struct blob_t {
	double x, y, strength, circularity, sign;
	friend std::ostream& operator<<(std::ostream& os, blob_t const& blob);
};

std::ostream& operator<<(std::ostream& os, blob_t const& blob);

using blobs_t = std::vector < blob_t > ;

//! Hessian blob detector
/*!
\param k_size gaussian kernel size
\return
*/
blobs_t detect_blobes(cv::Mat const& image, double const k_size = 1.2, double const threshold = 100., cv::Mat const& ignore_mask = cv::Mat());

//! Hessian ridge detector
/*!
\param k_size gaussian kernel size
\return
*/
//cv::Mat detect_ridges(cv::Mat const& image, double const k_size = 13, cv::Mat const& ignore_mask = cv::Mat(), derivatives_t& generate_derivatives = derivatives_t());

#endif

