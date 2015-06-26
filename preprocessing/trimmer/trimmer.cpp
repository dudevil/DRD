#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>

#include <opencv2\opencv.hpp>

#include <iputils\trim.hpp>


// Output width = sqrt(2.) * / 2 * input.width
cv::Mat get_inner_square(cv::Mat const& img) {
	assert(img.cols == img.rows);

	int const side_size = img.cols;
	float angle = 45;
	cv::Point rect_center(side_size / 2, side_size / 2);
	cv::Size rect_size(std::sqrt(2.) * side_size / 2., std::sqrt(2.) * side_size / 2.);

	cv::RotatedRect rect = cv::RotatedRect(rect_center, rect_size, angle);

	if (rect.angle < -45.) {
		angle += 90.0;
		cv::swap(rect_size.width, rect_size.height);
	}

	// get the rotation matrix
	cv::Mat M = getRotationMatrix2D(rect.center, angle, 1.0);
	// perform the affine transformation
	cv::Mat rotated, cropped;
	warpAffine(img, rotated, M, img.size(), cv::INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, rect.center, cropped);
	return cropped;
}

int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 5) {
		std::cout
			<< "usage:" << std::endl
			<< "./trimmer <input file> <output file> <output width> <trim_to_inner_square>" << std::endl
			<< "If <trim_to_inner_square> set to 1, result width = sqrt(2.) * / 2 * <output width>, image rotated by 45 degrees!" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename(argv[1]);
	std::string const output_filename(argv[2]);
	unsigned const output_side_size = std::stoi(argv[3]);
	bool take_inner_square = std::stoi(argv[4]);;

	// read image, resize, split
	cv::Mat3b const img = cv::imread(input_filename);

	cv::Mat3b img_trimmed = trim(img);

	// resize & write output
	cv::Mat3b img_resized;
	cv::resize(img_trimmed, img_resized, cv::Size(output_side_size, img_trimmed.rows * double(output_side_size) / img_trimmed.cols), 0, 0, cv::INTER_LANCZOS4);

	// make image square
	int const rows_remained = output_side_size - img_resized.rows;
	int const top = rows_remained / 2;
	int const bottom = rows_remained / 2 + rows_remained % 2;
	cv::copyMakeBorder(img_resized, img_resized, top, bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	img_resized = take_inner_square ? get_inner_square(img_resized) : img_resized;

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
