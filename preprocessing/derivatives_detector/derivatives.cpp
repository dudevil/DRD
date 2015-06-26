#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>


#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>


#include <iputils/derivatives.hpp>
#include <iputils/lcn.hpp>
#include <iputils/nonmaxsup.hpp>

#include <opencv2/imgproc/types_c.h>

float LoG(int x, int y, float sigma) {
	float xy = (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2));
	return -1.0 / (CV_PI * pow(sigma, 4)) * (1.0 - xy) * exp(-xy);
}

static cv::Mat LOGkernel(int size, float sigma) {
	cv::Mat kernel(size, size, CV_64F);
	int halfsize = size / 2;
	for (int x = -halfsize; x <= halfsize; ++x) {
		for (int y = -halfsize; y <= halfsize; ++y) {
			kernel.at<double>(x + halfsize, y + halfsize) = LoG(x, y, sigma);
		}
	}
	return kernel;
}

int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 5) {
		std::cout
			<< "usage:" << std::endl
			<< "./derivatives <input file> <output file> <kernel size> <as_gray>" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename("samples_1/32282_left.png");// argv[1]); //("samples/1100_left_1.png");// (argv[1]);
	std::string const output_filename("samples_1/32282_left_der.png");// argv[2]);
	unsigned const k_size = std::stoi(argv[3]); //5; // std::stoi(argv[3]);
	bool const cvt_to_gray = 1;// std::stoi(argv[4]);

	// read image
	cv::Mat img = cv::imread(input_filename);
	cv::Mat img_orig;
	img.convertTo(img_orig, CV_32FC3, 1.0 / 255.0);

	//cv::resize(img, img, cv::Size(512, 512), 0, 0, CV_INTER_CUBIC);

	// Split channels
	std::vector<cv::Mat> channels(img.channels());
	cv::split(img.clone(), channels);

	if (cvt_to_gray) {
		cv::cvtColor(img, img, CV_BGR2GRAY);
		img.convertTo(img, CV_32FC1, 1.0 / 255.0);
	} else {
		img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	}
	
	channels[1].convertTo(channels[1], CV_32FC3, 1.0 / 255.0);



	cv::Mat lcn_image = normalize_local_contrast(img, 35);
	cv::Mat lcn_image_green = normalize_local_contrast(channels[1], 35);
	//cv::GaussianBlur(lcn_image, lcn_image, cv::Size(5, 5), 0, 0);

	cv::Mat const kernel = -1*LOGkernel(11, 0.5);
	cv::Mat log_lcn, log_lcn_green;
	cv::filter2D(lcn_image, log_lcn, CV_32F, kernel);
	cv::filter2D(lcn_image_green, log_lcn_green, CV_32F, kernel);

	double const min_sigma = 1.7;
	cv::Mat normalized0 = detect_blobes(lcn_image, min_sigma);
	cv::Mat normalized1 = detect_blobes(lcn_image, 2 * min_sigma);
	cv::Mat normalized2 = detect_blobes(lcn_image, 3 * min_sigma);
	cv::Mat normalized3 = detect_blobes(lcn_image, 4 * min_sigma);
	cv::Mat normalized_green = detect_blobes(lcn_image_green, 1.7);


	//normalized.convertTo(normalized, CV_32F);
	normalized_green.convertTo(normalized_green, CV_32F);
	//cv::Mat normalized_thresh = normalized.clone();
	cv::Mat normalized_thresh_green = normalized_green.clone();
	cv::threshold(normalized_green, normalized_thresh_green, 3750.0, 0.0, CV_THRESH_TOZERO);

	cv::Mat maximums;
	nonMaximaSuppression(normalized_thresh_green, 7, maximums, cv::Mat());

	maximums.convertTo(maximums, CV_32F);
	cv::Mat lcn_maximums = maximums.mul(log_lcn_green);
	lcn_maximums.setTo(0, lcn_maximums < 0);

	cv::Mat3d result = img_orig.clone();

	std::vector<float> values;
	std::vector<cv::Point> coordinates;
	for (int i = 0; i < lcn_maximums.rows; ++i) {
		for (int j = 0; j < lcn_maximums.cols; ++j) {
			if (lcn_maximums.at<float>(j, i) > 1.0) {
				cv::circle(result, cv::Point(i*2, j*2), 10, cv::Scalar(0, 1, 0));
			}
		}
	}
	
	cv::imwrite(output_filename, result * 255.0);
	return EXIT_SUCCESS;
}
