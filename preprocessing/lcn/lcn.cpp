// This utility implements Local Contrast Normalization technique
// Description: http://bigwww.epfl.ch/sage/soft/localnormalization/

#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>


int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 5) {
		std::cout
			<< "usage:" << std::endl
			<< "./trimmer <input file> <output file> <kernel size> <as_gray>" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename(argv[1]);
	std::string const output_filename(argv[2]);
	unsigned const k_size = std::stoi(argv[3]);
	bool const cvt_to_gray = std::stoi(argv[4]);

	// read image
	cv::Mat img = cv::imread(input_filename);
	if (cvt_to_gray) {
		cv::cvtColor(img, img, CV_BGR2GRAY);
		img.convertTo(img, CV_64FC1, 1.0/255.0);
	} else {
		img.convertTo(img, CV_64FC3, 1.0/255.0);
	}
	
	// Smooth image and calculate difference
	cv::Mat img_smoothed;
	cv::GaussianBlur(img, img_smoothed, cv::Size(k_size, k_size), -1.0);
	cv::Mat diff = img_smoothed - img;

	// Smooth difference
	cv::Mat diff_smoothed;
	cv::GaussianBlur(diff.mul(diff), diff_smoothed, cv::Size(k_size, k_size), -1.0);
	cv::sqrt(diff_smoothed,diff_smoothed);

	// Calculate final result
	cv::Mat result = diff /   diff_smoothed;

	// Scale each channel to range [0, 1]
	std::vector<cv::Mat> channels(result.channels());
	cv::split(result, channels);

	std::for_each(channels.begin(), channels.end(), [](cv::Mat& ch){
		double min, max;
		cv::minMaxLoc(ch, &min, &max);
		ch = (ch - min)/(max-min);
	});
	cv::merge(channels, result);

	cv::imwrite(output_filename, result*255.0);

	return EXIT_SUCCESS;
}
