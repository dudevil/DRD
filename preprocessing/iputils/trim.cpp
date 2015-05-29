#include "trim.hpp"

#include <opencv2/imgproc/imgproc_c.h>

namespace {
	unsigned percentile(cv::Mat1b const& input, double const perc)
	{
		cv::Mat1b mat = input.clone();
		cv::resize(mat, mat, cv::Size(input.cols / 2, input.rows / 2));
		std::sort(mat.begin(), mat.end());
		return mat.at<unsigned char>(unsigned(perc*mat.size().area() / 100));
	}
}


//! Trim constant color image borders.
/*!
\param divider divide image size by this value to operate faster.
\return image with borders removed
*/
cv::Mat trim(cv::Mat const& image, unsigned const divider)
{
	cv::Mat3b img_small;
	cv::resize(image, img_small, cv::Size(image.cols / divider, image.rows / divider));

	cv::Mat1b b(img_small.size()), g(img_small.size()), r(img_small.size());
	std::vector<cv::Mat1b> channels{ b, g, r };

	cv::split(img_small, channels);

	// find region of interest
	unsigned const thresh_b = percentile(b, 25.0);
	unsigned const thresh_g = percentile(g, 25.0);
	unsigned const thresh_r = percentile(r, 25.0);
	cv::Mat1b mask = (b>thresh_b).mul((g>thresh_g).mul(r>thresh_r));

	// filter noise
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4)));

	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	// find largest contour and its bounding rectangle
	typedef std::vector<cv::Point> contour_t;
	std::vector<contour_t> contours;

	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::sort(contours.begin(), contours.end(), [](contour_t const& l, contour_t const& r){return cv::contourArea(l) > cv::contourArea(r); });

	cv::Rect bounding_box = cv::boundingRect(contours[0]);

	// Enlarge bounding rect and crop image
	cv::Rect real_bounding_box(bounding_box.x*divider, bounding_box.y*divider, bounding_box.width*divider, bounding_box.height*divider);
	cv::Mat cropped = image(real_bounding_box);

	return cropped;
}

