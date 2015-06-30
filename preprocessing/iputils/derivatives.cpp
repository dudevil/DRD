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

	const size_t derivatives_ksize = 55;
}

derivatives_t generate_derivatives(cv::Mat const& image, double const sigma = 1.7)
{
	CV_Assert(sigma >= 0.9);
	CV_Assert(image.depth() == CV_32FC1 || image.depth() == CV_32FC3);

	cv::Mat1f gaussian(derivatives_ksize, 1, 0.0);
	for (int i = 0; i < derivatives_ksize; ++i) {
		gaussian.at<float>(i) = std::exp(
			-1. * 
			std::pow(double(i - (derivatives_ksize - 1) / 2.), 2.) /
			(2 * std::pow(sigma, 2.))
			);
	}
	cv::mulTransposed(gaussian, gaussian, false);
	
	// result has same depth as input
	cv::Mat dxx, dxy, dyy;
	cv::Sobel(gaussian, dxx, -1, 2, 0, 3, 1., 0., cv::BORDER_REPLICATE);
	cv::Sobel(gaussian, dxy, -1, 1, 1, 3, 1., 0., cv::BORDER_REPLICATE);
	cv::Sobel(gaussian, dyy, -1, 0, 2, 3, 1., 0., cv::BORDER_REPLICATE);

	derivatives_t result{ dxx.clone(), dxy.clone(), dyy.clone() };
	cv::filter2D(image, result[0], image.depth(), dxx, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	cv::filter2D(image, result[1], image.depth(), dxy, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	cv::filter2D(image, result[2], image.depth(), dyy, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

	return result;
}

cv::Mat detect_blobes(cv::Mat const& image, double const sigma, cv::Mat const& ignore_mask, derivatives_t& _derivatives)
{
	if (_derivatives.empty()) {
		_derivatives = generate_derivatives(image, sigma);
	};
	cv::Mat const& dxx = _derivatives[0];
	cv::Mat const& dxy = _derivatives[1];
	cv::Mat const& dyy = _derivatives[2];

	// normalization
	cv::Mat result = (dxx.mul(dyy) - dxy.mul(dxy)) * std::pow(sigma, 0.5); // / std::pow(sigma, 2.);
	cv::Mat trace = ((dxx + dyy) < 0.0);
	trace.convertTo(trace, result.type(), 1. / 255., 0);
	result = cv::abs(result).mul(trace);

	if (!ignore_mask.empty()) {
		result.setTo(cv::Scalar::all(0), ~ignore_mask);
	}
	
	return result; // scale_to_range();
}

cv::Mat detect_ridges(cv::Mat const& image, double const sigma, cv::Mat const& ignore_mask, derivatives_t& _derivatives)
{
	if (_derivatives.empty()) {
		_derivatives = generate_derivatives(image, sigma);
	};
	cv::Mat const& dxx = _derivatives[0];
	cv::Mat const& dxy = _derivatives[1];
	cv::Mat const& dyy = _derivatives[2];

	cv::Mat root;
	cv::sqrt((dxx - dyy).mul(dxx - dyy) + 4 * dxy.mul(dxy), root);
	cv::Mat result = root - dxx - dyy;

	return result;// scale_to_range();
}