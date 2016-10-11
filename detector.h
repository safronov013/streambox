#ifndef __INCLUDED_DETECTOR_H__
#define __INCLUDED_DETECTOR_H__

#pragma comment( lib, "detector.lib" )

#include <iostream>
#include <future>
// #include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/imgproc/imgproc_c.h>
// #include <opencv2/video/tracking.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <tesseract/baseapi.h>

#pragma comment( lib, "opencv_core2413.lib" )
#pragma comment( lib, "opencv_highgui2413.lib" )
#pragma comment( lib, "opencv_imgproc2413.lib" )
#pragma comment( lib, "opencv_video2413.lib" )
#pragma comment( lib, "tesseract/libtesseract302.lib" )

#define OCR_MAX 16

typedef cv::Vec3b pixel_t;
typedef std::vector<cv::Rect> places_t;
typedef std::future<places_t> future_t;

enum algorithm_t
{
	ALGO_DIFF=1,
	ALGO_CURRENT
};

void tesseract_init();
void img_detect_label( cv::Mat& frame_curr );
void img_draw_rect( cv::Mat& frame_curr );

#endif
