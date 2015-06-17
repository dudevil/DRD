#include "nonmaxsup.hpp"

void nonMaximaSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, const cv::Mat mask) {

	// initialise the block mask and destination
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	cv::Mat block = 255 * cv::Mat_<uint8_t>::ones(cv::Size(2 * sz + 1, 2 * sz + 1));
	dst = cv::Mat_<uint8_t>::zeros(src.size());

	// iterate over image blocks
	for (int m = 0; m < M; m += sz + 1) {
		for (int n = 0; n < N; n += sz + 1) {
			cv::Point  ijmax;
			double vcmax, vnmax;

			// get the maximal candidate within the block
			cv::Range ic(m, std::min(m + sz + 1, M));
			cv::Range jc(n, std::min(n + sz + 1, N));
			cv::minMaxLoc(src(ic, jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic, jc) : cv::noArray());
			cv::Point cc = ijmax + cv::Point(jc.start, ic.start);

			// search the neighbours centered around the candidate for the true maxima
			cv::Range in(std::max(cc.y - sz, 0), std::min(cc.y + sz + 1, M));
			cv::Range jn(std::max(cc.x - sz, 0), std::min(cc.x + sz + 1, N));

			// mask out the block whose maxima we already know
			cv::Mat_<uint8_t> blockmask;
			block(cv::Range(0, in.size()), cv::Range(0, jn.size())).copyTo(blockmask);
			cv::Range iis(ic.start - in.start, std::min(ic.start - in.start + sz + 1, in.size()));
			cv::Range jis(jc.start - jn.start, std::min(jc.start - jn.start + sz + 1, jn.size()));
			blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(), iis.size()));

			minMaxLoc(src(in, jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in, jn).mul(blockmask) : blockmask);
			cv::Point cn = ijmax + cv::Point(jn.start, in.start);

			// if the block centre is also the neighbour centre, then it's a local maxima
			if (vcmax > vnmax) {
				dst.at<uint8_t>(cc.y, cc.x) = 255;
			}
		}
	}
}
