#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <iputils/lcn.hpp>
#include <opencv2/imgproc/types_c.h>

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
		img.convertTo(img, CV_64FC1, 1.0 / 255.0);
	} else {
		img.convertTo(img, CV_64FC3, 1.0 / 255.0);
	}
	
	cv::Mat normalized = normalize_local_contrast(img, k_size);

	cv::imwrite(output_filename, normalized * 255.0);
	return EXIT_SUCCESS;
}
