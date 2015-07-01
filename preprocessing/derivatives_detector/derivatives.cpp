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

#include <thread>

template<typename Iterator, class Function>
void parallel_for(const Iterator& first, const Iterator& last, Function&& f, const int nthreads = 1, const int threshold = 1000)
{
	const unsigned int group = std::max(std::max(ptrdiff_t(1), ptrdiff_t(std::abs(threshold))), ((last - first)) / std::abs(nthreads));
	std::vector<std::thread> threads;
	threads.reserve(nthreads);
	Iterator it = first;
	for (; it < last - group; it += group) {
		threads.push_back(std::thread([=, &f](){std::for_each(it, std::min(it + group, last), f); }));
	}
	std::for_each(it, last, f); // last steps while we wait for other threads

	std::for_each(threads.begin(), threads.end(), [](std::thread& x){x.join(); });
}

int main(int const argc, char const* const argv[])
{
	// read command line
	if (argc != 5) {
		std::cout
			<< "" << std::endl
			<< "" << std::endl;
		return EXIT_FAILURE;
	}
	std::string const input_filename(argv[1]);
	std::string const output_filename(argv[2]);
	double const thresh = std::stod(argv[3]); // 450.
	int const points_count = std::stoi(argv[4]); // 450.
	// read image
	cv::Mat const img_color = cv::imread(input_filename);
	cv::Mat img_gray;
	cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);
	img_gray.convertTo(img_gray, CV_32FC1, 1. / 255.);

	blobs_t blobs = detect_blobes(img_gray, 1.2, 400.);

	std::ofstream blobs_list(output_filename + ".csv");
	std::copy(blobs.cbegin(), blobs.cend(), std::ostream_iterator<blob_t>(blobs_list, "\n"));

	cv::Mat result = img_color.clone();

	std::for_each(blobs.begin(), blobs.end(), [&result](blob_t const& b) {
		cv::circle(result, cv::Point(b.x, b.y), std::abs(b.strength) * 0.025, cv::Scalar((b.sign < 0 ? 255 : 0), 255, 0));
	});

	cv::imwrite(output_filename, result);
	return EXIT_SUCCESS;
}
