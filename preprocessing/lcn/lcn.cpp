#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <iputils/lcn.hpp>
#include <opencv2/imgproc/types_c.h>



cv::Mat apply_lcn(cv::Mat const& img, int const ksize, bool const normalize_global)
{
	cv::Mat const normalized = normalize_local_contrast(img, ksize);
	cv::Mat const globally_normalized = normalize_global ? normalize_global_contrast(normalized) : normalized;
	return scale_min_max(globally_normalized);
}

// LCN on RGB channel independent
cv::Mat lcn_color(cv::Mat const& img, int const ksize, bool const normalize_global)
{
	assert(!img.empty() && img.channels() == 3);
	assert(ksize > 0);
	cv::Mat img_converted;
	img.convertTo(img_converted, CV_32FC3, 1.0 / 255.0);
	return apply_lcn(img_converted, ksize, normalize_global) * 255.0;
}

cv::Mat lcn_gray(cv::Mat const& img, int const ksize, bool const normalize_global)
{
	assert(!img.empty() && img.channels() == 1);
	assert(ksize > 0);
	cv::Mat img_converted;
	cv::cvtColor(img, img_converted, CV_BGR2GRAY);
	img_converted.convertTo(img_converted, CV_32FC1, 1.0 / 255.0);
	return apply_lcn(img_converted, ksize, normalize_global) * 255.0;
}

cv::Mat lcn_yuv(cv::Mat const& img, int const ksize, bool const normalize_global)
{
	assert(!img.empty() && img.channels() == 3);
	assert(ksize > 0);

	cv::Mat img_yuv;
	cv::cvtColor(img, img_yuv, CV_BGR2YUV);
	img_yuv.convertTo(img_yuv, CV_32FC3);

	// Split channels
	std::vector<cv::Mat> channels(img_yuv.channels());
	cv::split(img_yuv, channels);

	// LCN on Y channel
	channels[0] = normalize_local_contrast(channels[0] / 255.0, ksize);
	
	// GCN on Y channel if requested
	channels[0] = normalize_global ? normalize_global_contrast(channels[0]) : channels[0];

	// Y channel now has unit variance, so scale it to [0, 1] range
	channels[0] += gaussian_99_percentile;
	channels[0].convertTo(channels[0], channels[0].type(), 255. / (2. * gaussian_99_percentile));

	cv::merge(channels, img_yuv);

	//img_normed1.convertTo(img_normed1, CV_8U)
	cv::Mat result;
	img_yuv.convertTo(img_yuv, CV_8UC3);
	cv::cvtColor(img_yuv, result, CV_YUV2BGR);
	return result;
}

static const cv::String keys =
	"{in      |       | input                             }"
	"{out    |       | output                            }"
	"{ksize   | 21    | kernel size                       }"
	"{gray    | false | convert to gray before processing }"
	"{gcn     | false | do global contrast normalization  }"
	"{yuv     | false | operate on Y of YUV               }"
	;

int main(int const argc, char const* const argv[])
{
	cv::CommandLineParser cli(argc, argv, keys);
	if (!cli.check()) {
		cli.printErrors();
		cli.printMessage();
		return EXIT_FAILURE;
	}

	// read command line
	std::string const input_filename  = cli.get<std::string>("in");
	std::string const output_filename = cli.get<std::string>("out");
	int const k_size             = cli.get<int>("ksize");

	if (input_filename.empty() || output_filename.empty() || k_size < 0) {
		cli.printMessage();
		return EXIT_FAILURE;
	}

	bool const cvt_to_gray            = cli.get<bool>("gray");
	bool const normalize_global       = cli.get<bool>("gcn");
	bool const operate_in_yuv         = cli.get<bool>("yuv");

	// read image
	cv::Mat const img = cv::imread(input_filename);

	auto const do_lcn = operate_in_yuv ? lcn_yuv : (cvt_to_gray ? lcn_gray : lcn_color);

	cv::Mat const result = do_lcn(img, k_size, normalize_global);

	cv::imwrite(output_filename, result);
	return EXIT_SUCCESS;
}
