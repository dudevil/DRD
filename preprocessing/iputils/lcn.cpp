#include "lcn.hpp"

#include <assert.h>

const double gaussian_99_percentile = 2.326347874040841100885606163346911723351817141532013069065640247890876626456034487356822930790951316943;

cv::Mat scale_min_max(cv::Mat const& img)
{
	// Scale each channel to range [0, 1]
	std::vector<cv::Mat> channels(img.channels());
	cv::split(img, channels);
	
	std::for_each(channels.begin(), channels.end(), [](cv::Mat& ch){
		double min, max;
		cv::minMaxLoc(ch, &min, &max);
		ch = (ch - min) / (max - min);
	});

	cv::Mat result;
	cv::merge(channels, result);
	return result;
}

cv::Mat normalize_local_contrast(cv::Mat const& image, int const k_size)
{
	assert(!image.empty());
	assert(k_size > 0);

	// Smooth image and calculate difference
	cv::Mat img_smoothed;
	cv::GaussianBlur(image, img_smoothed, cv::Size(k_size, k_size), -1.0);
	cv::Mat diff = img_smoothed - image;

	// Smooth difference
	cv::Mat diff_smoothed;
	cv::GaussianBlur(diff.mul(diff), diff_smoothed, cv::Size(k_size, k_size), -1.0);
	cv::sqrt(diff_smoothed, diff_smoothed);

	// Calculate final result
	cv::Mat result = diff / diff_smoothed;
	return result;
}

cv::Mat normalize_global_contrast(cv::Mat const& image)
{
	assert(image.channels() == 1);
	cv::Scalar mean, stddev;
	cv::meanStdDev(image, mean, stddev);
	
	cv::Mat result;
	image -= mean[0];
	image.convertTo(result, image.type(), 1.0 / stddev[0]);
	return result;
}

