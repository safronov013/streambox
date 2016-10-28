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
#include "queue_manager.h"


#pragma comment( lib, "opencv_core2413.lib" )
#pragma comment( lib, "opencv_highgui2413.lib" )
#pragma comment( lib, "opencv_imgproc2413.lib" )
#pragma comment( lib, "opencv_video2413.lib" )
#pragma comment( lib, "tesseract/libtesseract302.lib" )

#define OCR_MAX 8

typedef cv::Vec3b pixel_t;
typedef std::vector<cv::Rect> places_t;
typedef std::future<places_t> future_t;

enum algorithm_t
{
	ALGO_DIFF=1,
	ALGO_CURRENT
};

typedef struct _GPUVARS_S
{
	cv::gpu::GpuMat frame_curr;
	cv::gpu::GpuMat frame_curr_gray;
	cv::gpu::GpuMat frame_prev_gray;
	cv::Mat img_curr_gray;
} GPUVARS;

class ImgParams
{
public:
	int threshold = 0;
	int dilate = 0;
	int missed = 10;
	int reccurence = 1;
	cv::Size dilate_kernel_size{3, 1};
	cv::Size erode_kernel_size{3, 3};
	cv::Mat dilate_kernel;
	cv::Mat erode_kernel;
	algorithm_t algo_t = ALGO_DIFF;
	cv::Mat img;
	cv::Mat img_threshold;
	cv::Mat img_dilated;
	cv::Mat img_eroded;
	cv::gpu::GpuMat gpu_img;
	cv::gpu::GpuMat gpu_img_threshold;
	cv::gpu::GpuMat gpu_blacked_prev;
	cv::gpu::GpuMat gpu_blacked_curr;
	cv::gpu::GpuMat gpu_img_diff;
	cv::gpu::GpuMat gpu_img_bitand;
	cv::gpu::GpuMat gpu_img_eroded;
	cv::gpu::GpuMat gpu_img_dilated;
	std::vector<cv::Rect> regions;
	std::vector<std::vector<cv::Point>> contours;
	places_t places;
	cv::Rect roi_place;
	ImgParams();
	ImgParams( const int t, const int d, const cv::Size& d_kernel, const cv::Size& e_kernel, algorithm_t a );
	friend std::ostream& operator<< ( std::ostream& ostr, const ImgParams& param );
};

void tesseract_init();
// void img_detect_label( cv::Mat& frame_curr, GPUVARS& g );
// void img_detect_label( cv::Mat& frame_curr, std::vector<ImgParams>& params, GPUVARS* g, QueueManager& worker );
void img_detect_label( cv::Mat& frame_curr, std::vector<ImgParams>& params, GPUVARS* g );
void img_draw_rect( cv::Mat& frame_curr );
bool identify_text( const cv::Mat& img, tesseract::TessBaseAPI& ocr_item );
void roi_normalize( cv::Rect& roi, int width, int height );

#endif
