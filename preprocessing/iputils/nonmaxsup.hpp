/*
*  Software License Agreement (BSD License)
*
*  Copyright (c) 2012, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  File:    nms.cpp
*  Author:  Hilton Bristow
*  Created: Jul 19, 2012
*/

#include <stdio.h>
#include <iostream>
#include <limits>

#include <stdint.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*! @brief suppress non-maximal values
*
* nonMaximaSuppression produces a mask (dst) such that every non-zero
* value of the mask corresponds to a local maxima of src. The criteria
* for local maxima is as follows:
*
* 	For every possible (sz x sz) region within src, an element is a
* 	local maxima of src iff it is strictly greater than all other elements
* 	of windows which intersect the given element
*
* Intuitively, this means that all maxima must be at least sz+1 pixels
* apart, though the spacing may be greater
*
* A gradient image or a constant image has no local maxima by the definition
* given above
*
* The method is derived from the following paper:
* A. Neubeck and L. Van Gool. "Efficient Non-Maximum Suppression," ICPR 2006
*
* Example:
* \code
* 	// create a random test image
* 	Mat random(Size(2000,2000), DataType<float>::type);
* 	randn(random, 1, 1);
*
* 	// only look for local maxima above the value of 1
* 	Mat mask = (random > 1);
*
* 	// find the local maxima with a window of 50
* 	Mat maxima;
* 	nonMaximaSuppression(random, 50, maxima, mask);
*
* 	// optionally set all non-maxima to zero
* 	random.setTo(0, maxima == 0);
* \endcode
*
* @param src the input image/matrix, of any valid cv type
* @param sz the size of the window
* @param dst the mask of type CV_8U, where non-zero elements correspond to
* local maxima of the src
* @param mask an input mask to skip particular elements
*/
void nonMaximaSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, cv::Mat const& mask = cv::Mat());
