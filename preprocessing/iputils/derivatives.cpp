#include "derivatives.hpp"

namespace {

	// Scale to range [0, 1]
	cv::Mat scale_to_range(cv::Mat& mat)
	{
		std::vector<cv::Mat> channels(mat.channels());
		cv::split(mat, channels);

		std::for_each(channels.begin(), channels.end(), [](cv::Mat& ch){
			double min, max;
			cv::minMaxLoc(ch, &min, &max);
			ch = (ch - min) / (max - min);
		});
		cv::merge(channels, mat);
		return mat;
	}
}

derivatives_t generate_derivatives(cv::Mat const& image, double const k_size = 13)
{
	CV_Assert(image.depth() == CV_64FC1 || image.depth() == CV_64FC3);
	cv::Mat dxx, dxy, dyy;
	derivatives_t result{ dxx, dxy, dyy };
	// result has same depth as input
	cv::Sobel(image, result[0], -1, 2, 0, k_size);
	cv::Sobel(image, result[1], -1, 1, 1, k_size);
	cv::Sobel(image, result[2], -1, 0, 2, k_size);
	return result;
}

cv::Mat detect_blobes(cv::Mat const& image, double const k_size, cv::Mat const& ignore_mask, derivatives_t& _derivatives)
{
	if (_derivatives.empty()) {
		_derivatives = generate_derivatives(image, k_size);
	};
	cv::Mat const& dxx = _derivatives[0];
	cv::Mat const& dxy = _derivatives[1];
	cv::Mat const& dyy = _derivatives[2];

	cv::Mat result = dxx.mul(dyy) - dxy.mul(dxy);
	
	if (!ignore_mask.empty()) {
		result.setTo(cv::Scalar::all(0), ~ignore_mask);
	}
	
	return result; // scale_to_range();
}

cv::Mat detect_ridges(cv::Mat const& image, double const k_size, cv::Mat const& ignore_mask, derivatives_t& _derivatives)
{
	if (_derivatives.empty()) {
		_derivatives = generate_derivatives(image, k_size);
	};
	cv::Mat const& dxx = _derivatives[0];
	cv::Mat const& dxy = _derivatives[1];
	cv::Mat const& dyy = _derivatives[2];

	cv::Mat root;
	cv::sqrt((dxx - dyy).mul(dxx - dyy) + 4 * dxy.mul(dxy), root);
	cv::Mat result = root - dxx - dyy;

	return scale_to_range(result);
}