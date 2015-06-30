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

	cv::Mat lcn_image = normalize_local_contrast(img_gray, 35);

	auto say_max = [] (cv::Mat const& m) {
		double min, max;
		cv::minMaxLoc(m, &min, &max);
		std::cout << "min: " << min << ", max: " << max << std::endl;
		return m;
	};

	double const min_sigma = 1.2;
	cv::Mat const normalized0 = say_max(detect_blobes(lcn_image, 1.0 * min_sigma));
	cv::Mat const normalized1 = say_max(detect_blobes(lcn_image, 2.0 * min_sigma));
	cv::Mat const normalized2 = say_max(detect_blobes(lcn_image, 3.0 * min_sigma));
	cv::Mat const normalized3 = say_max(detect_blobes(lcn_image, 4.0 * min_sigma));
	cv::Mat normalized0nonmaxsup = cv::Mat::zeros(lcn_image.size(), lcn_image.type());
	cv::Mat normalized1nonmaxsup = cv::Mat::zeros(lcn_image.size(), lcn_image.type());
	cv::Mat normalized2nonmaxsup = cv::Mat::zeros(lcn_image.size(), lcn_image.type());
	cv::Mat normalized3nonmaxsup = cv::Mat::zeros(lcn_image.size(), lcn_image.type());

	auto detect_local_maximas = [&](cv::Mat & current, cv::Mat & prev, cv::Mat & next, cv::Mat & dest){
		for (int i = 1; i < lcn_image.rows - 1; ++i) {
			for (int j = 1; j < lcn_image.cols - 1; ++j) {
				cv::Point loc(j, i);
				if (is_local_max(current, prev, next, loc)) {
					dest.at<float>(loc) = current.at<float>(loc);
				}
			}
		}
	};

	struct Args { cv::Mat cur, prev, next, dest; };
	std::vector<Args> detect_calls = {
		{ normalized0, cv::Mat(), normalized1, normalized0nonmaxsup },
		{ normalized1, normalized0, normalized2, normalized1nonmaxsup },
		{ normalized2, normalized1, normalized3, normalized2nonmaxsup },
		{ normalized3, normalized2, cv::Mat(), normalized3nonmaxsup }
	};

	parallel_for(detect_calls.begin(), detect_calls.end(), [&](Args& arg)
	{
		detect_local_maximas(arg.cur, arg.prev, arg.next, arg.dest);
	}, 4);

	int candidates_count = 0;
	double effective_thresh = thresh;
	cv::Mat maximums;
	while (candidates_count < points_count) {
		cv::Mat normalized0nonmaxsup_thresh;
		cv::Mat normalized1nonmaxsup_thresh;
		cv::Mat normalized2nonmaxsup_thresh;
		cv::Mat normalized3nonmaxsup_thresh;

		cv::threshold(normalized0nonmaxsup, normalized0nonmaxsup_thresh, effective_thresh, 0.0, CV_THRESH_TOZERO);
		cv::threshold(normalized1nonmaxsup, normalized1nonmaxsup_thresh, effective_thresh, 0.0, CV_THRESH_TOZERO);
		cv::threshold(normalized2nonmaxsup, normalized2nonmaxsup_thresh, effective_thresh, 0.0, CV_THRESH_TOZERO);
		cv::threshold(normalized3nonmaxsup, normalized3nonmaxsup_thresh, effective_thresh, 0.0, CV_THRESH_TOZERO);
		maximums = normalized0nonmaxsup_thresh + normalized1nonmaxsup_thresh + normalized2nonmaxsup_thresh + normalized3nonmaxsup_thresh;
		effective_thresh *= 0.9;
		candidates_count = cv::countNonZero(maximums);
	}

	cv::Mat result = img_color.clone();
	for (int i = 1; i < lcn_image.rows - 1; ++i) {
		for (int j = 1; j < lcn_image.cols - 1; ++j) {
			cv::Point loc(j, i);
			if (maximums.at<float>(loc) > 0.0) {
				cv::circle(result, loc, 10, cv::Scalar(255, 255, 0), 1);
			}
		}
	}

	cv::imwrite(output_filename, result);
	return EXIT_SUCCESS;
}
