#include "lcn.hpp"

cv::Mat normalize_local_contrast(cv::Mat const& image, double const k_size)
{
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

	// Scale each channel to range [0, 1]
	std::vector<cv::Mat> channels(result.channels());
	cv::split(result, channels);

	std::for_each(channels.begin(), channels.end(), [](cv::Mat& ch){
		double min, max;
		cv::minMaxLoc(ch, &min, &max);
		ch = (ch - min) / (max - min);
	});
	cv::merge(channels, result);
	return result;
}

