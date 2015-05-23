#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2\opencv.hpp>

#include <iputils\trim.hpp>

int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 4) {
		std::cout
			<< "usage:" << std::endl
			<< "./trimmer <input file> <output file> <output width>" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename(argv[1]);
	std::string const output_filename(argv[2]);
	unsigned const output_side_size = std::stoi(argv[3]);

	// read image, resize, split
	cv::Mat3b const img = cv::imread(input_filename);

	cv::Mat3b img_trimmed = trim(img);

	// resize & write output
	cv::Mat3b img_resized;
	cv::resize(img_trimmed, img_resized, cv::Size(output_side_size, img_trimmed.rows * double(output_side_size) / img_trimmed.cols));

	cv::imwrite(output_filename, img_resized);

	return EXIT_SUCCESS;
}

#if 0
	cv::Mat1b img_gray;
	cv::cvtColor(img_small, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat roww = img_gray(cv::Rect(0, img_gray.rows/2, img_gray.cols, 1));

	cv::Mat1d sobel;
	cv::Sobel(roww, sobel, CV_64F, 1, 0, 15);
	sobel = cv::abs(sobel);

	double min,max;
	int min_ind, max_ind;
	cv::minMaxIdx(sobel, &min, &max, &min_ind, &max_ind);
	sobel.colRange(max_ind-5, max_ind+5) = 0.0;

	cv::minMaxIdx(sobel, nullptr, &max, nullptr, &max_ind);
	sobel.colRange(max_ind-5, max_ind+5) = 0.0;
#endif
