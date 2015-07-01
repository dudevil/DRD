#include "derivatives.hpp"

#include "lcn.hpp"

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

	bool is_local_max(cv::Mat const& current, cv::Mat const& prev, cv::Mat const& next, cv::Point const& loc)
	{
		bool const max_in_current =
			current.at<float>(loc) > current.at<float>(loc + cv::Point(-1, -1)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(-1, 0)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(-1, 1)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(0, -1)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(0, 1)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(1, -1)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(1, 0)) &&
			current.at<float>(loc) > current.at<float>(loc + cv::Point(1, 1));
		bool const max_in_prev = prev.empty() ? true :
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(-1, -1)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(-1, 0)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(-1, 1)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(0, -1)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(0, 0)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(0, 1)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(1, -1)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(1, 0)) &&
			current.at<float>(loc) > prev.at<float>(loc + cv::Point(1, 1));
		bool const max_in_next = next.empty() ? true :
			current.at<float>(loc) > next.at<float>(loc + cv::Point(-1, -1)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(-1, 0)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(-1, 1)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(0, -1)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(0, 0)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(0, 1)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(1, -1)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(1, 0)) &&
			current.at<float>(loc) > next.at<float>(loc + cv::Point(1, 1));
		return max_in_current && max_in_prev & max_in_next;
	}

	void say_range(cv::Mat const& m)
	{
		double min, max;
		cv::minMaxLoc(m, &min, &max);
		std::cout << "min: " << min << ", max: " << max << std::endl;
	};

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

	cv::Mat derivative_xx, derivative_xy, derivative_yy;
	cv::filter2D(image, derivative_xx, image.depth(), dxx, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	cv::filter2D(image, derivative_xy, image.depth(), dxy, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	cv::filter2D(image, derivative_yy, image.depth(), dyy, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

	return { derivative_xx, derivative_xy, derivative_yy, sigma };
}

std::ostream& operator<<(std::ostream& os, blob_t const& blob)
{
	return os << blob.x << "," << blob.y << "," << blob.strength;
}

blobs_t detect_blobes(cv::Mat const& image, double const sigma, double const threshold, cv::Mat const& ignore_mask)
{
	cv::Mat lcn_image = normalize_local_contrast(image, 35);
	std::vector<derivatives_t> derivatives = {
		generate_derivatives(lcn_image, 1.0 * sigma),
		generate_derivatives(lcn_image, 2.0 * sigma),
		generate_derivatives(lcn_image, 3.0 * sigma),
		generate_derivatives(lcn_image, 4.0 * sigma)
	};

	// normalization
	std::vector<cv::Mat> det_of_hessian(derivatives.size());
	std::transform(derivatives.cbegin(), derivatives.cend(), det_of_hessian.begin(), [](derivatives_t const& d) {
		return (d.dxx.mul(d.dyy) - d.dxy.mul(d.dxy)) * std::pow(d.sigma, 0.5);
	});
	
	std::vector<cv::Mat> trace(derivatives.size());
	std::transform(derivatives.cbegin(), derivatives.cend(), trace.begin(), [](derivatives_t const& d) {
		return (d.dxx + d.dyy);
	});
	
	std::vector<cv::Mat> circularity(derivatives.size());
	std::transform(det_of_hessian.cbegin(), det_of_hessian.cend(), trace.cbegin(), circularity.begin(), [](cv::Mat const& doh, cv::Mat const& tr) {
		cv::Mat const disciminato = tr - 4 * doh;
		cv::sqrt(disciminato, disciminato);
		cv::Mat h1 = (-tr + disciminato)/2.;
		cv::Mat h2 = (-tr - disciminato)/2.;
		return cv::abs(cv::min(h1, h2) / cv::max(h1, h2));
	});

	std::for_each(det_of_hessian.cbegin(), det_of_hessian.cend(), say_range);
	std::for_each(det_of_hessian.begin(), det_of_hessian.end(), [](cv::Mat& doh){doh.setTo(0.0, doh < 0.0); });

	blobs_t detected_blobs;
	for (int i = 1; i < lcn_image.rows - 1; ++i) {
		for (int j = 1; j < lcn_image.cols - 1; ++j) {
			cv::Point loc(j, i);

			for (int s = 0; s < det_of_hessian.size(); ++s) {
				auto const& prev = (s - 1) < 0 ? cv::Mat() : det_of_hessian[s - 1];
				auto const& next = (s + 1) >= det_of_hessian.size() ? cv::Mat() : det_of_hessian[s + 1];
				bool const local_max = is_local_max(det_of_hessian[s], prev, next, loc);
				if (local_max && det_of_hessian[s].at<float>(loc) > threshold) {
					//double x, y, strength, e1, e2, sign;
					detected_blobs.push_back(blob_t{ loc.x, loc.y, det_of_hessian[s].at<float>(loc), circularity[s].at<float>(loc), trace[s].at<float>(loc) });
				}
			}
		}
	}
	return detected_blobs;
}

#if 0
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
#endif
