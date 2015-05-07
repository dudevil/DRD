#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

unsigned percentile(cv::Mat1b const& input, double const perc)
{
	cv::Mat1b mat = input.clone();
	cv::resize(mat,mat, cv::Size(input.cols/2,input.rows/2));
	std::sort(mat.begin(), mat.end());
	return mat.at<unsigned char>(unsigned(perc*mat.size().area()/100));
}

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
	unsigned const divider = 8;

	cv::Mat3b img_small;
	cv::resize(img, img_small, cv::Size(img.cols/divider,img.rows/divider));

	cv::Mat1b b(img_small.size()),g(img_small.size()),r(img_small.size());
	std::vector<cv::Mat1b> channels(3);
	channels[0] = b; // Sorry guys, my VS2012 doesn't support initializer lists
	channels[1] = g;
	channels[2] = r;

	cv::split(img_small, channels);

	// find region of interest
	unsigned const thresh_b = percentile(b, 25.0);
	unsigned const thresh_g = percentile(g, 25.0);
	unsigned const thresh_r = percentile(r, 25.0);
	cv::Mat1b mask = (b>thresh_b).mul((g>thresh_g).mul(r>thresh_r));

	// filter noise
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4)));

	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

	// find largest contour and its bounding rectangle
	typedef std::vector<cv::Point> contour_t;
	std::vector<contour_t> contours;

	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::sort(contours.begin(), contours.end(), [](contour_t const& l, contour_t const& r){return cv::contourArea(l) > cv::contourArea(r);});

	cv::Rect bounding_box = cv::boundingRect(contours[0]);
	
	// Enlarge bounding rect and crop image
	cv::Rect real_bounding_box(bounding_box.x*divider,bounding_box.y*divider,bounding_box.width*divider,bounding_box.height*divider);
	cv::Mat cropped = img(real_bounding_box);
	
	// resize & write output
	cv::resize(cropped, cropped, cv::Size(output_side_size,cropped.rows * double(output_side_size)/cropped.cols));

	cv::imwrite(output_filename, cropped);

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
