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

int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 5) {
		std::cout
			<< "usage:" << std::endl
			<< "./derivatives <input file> <output file> <kernel size> <as_gray>" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename(argv[1]); //("samples/1100_left_1.png");// (argv[1]);
	std::string const output_filename(argv[2]);
	unsigned const k_size = std::stoi(argv[3]); //5; // std::stoi(argv[3]);
	bool const cvt_to_gray = 1;// std::stoi(argv[4]);

	// read image
	cv::Mat img = cv::imread(input_filename);

	cv::resize(img, img, cv::Size(512, 512), 0, 0, CV_INTER_CUBIC);

	if (cvt_to_gray) {
		cv::cvtColor(img, img, CV_BGR2GRAY);
		img.convertTo(img, CV_64FC1, 1.0 / 255.0);
	} else {
		img.convertTo(img, CV_64FC3, 1.0 / 255.0);
	}
	
	cv::Mat mask;
	cv::inRange(img, cv::Scalar(0.05, 0.05, 0.05), cv::Scalar(1, 1, 1), mask);
	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));


	cv::Mat lcn_image = normalize_local_contrast(img, 25);

	//cv::GaussianBlur(lcn_image, lcn_image, cv::Size(5, 5), 0, 0);

	cv::Mat normalized = detect_blobes(lcn_image, 7, mask);
	normalized.convertTo(normalized, CV_32F);
	cv::Mat normalized_thresh;
	cv::threshold(normalized, normalized_thresh, 3500.0, 0.0, CV_THRESH_TOZERO);

	cv::Mat1b maximums;
	nonMaximaSuppression(normalized_thresh, 7, maximums, cv::Mat());

	//maximum.values
	std::vector<float> values;
	std::vector<cv::Point> coordinates;
	for (cv::Mat1b::iterator it = maximums.begin(); it != maximums.end(); ++it){
		if (*it) {
			values.push_back(normalized.at<float>(it.pos()));
			coordinates.push_back(it.pos());
		}
	}
	
	std::ofstream os(input_filename + ".csv");
	for (int i = 0; i < values.size(); ++i) {
		os
			<< coordinates[i].x << ","
			<< coordinates[i].y << ","
			<< values[i] << std::endl;
	}

	//cv::imshow("input", img);
	//cv::imshow("mask", mask);
	//cv::imshow("result", normalized.setTo(0, 1-mask));
	//cv::waitKey(0);
	//cv::imwrite(output_filename, normalized * 255.0);
	return EXIT_SUCCESS;
}
