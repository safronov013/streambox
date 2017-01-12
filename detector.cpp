#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
#include <iterator>
#include <future>
#include "detector.h"


int frame_cnt = 0;
extern QueueManager worker;
std::deque<ROI_RESULT> g_results;
std::mutex g_mutex;


ImgParams::ImgParams()
{

}

ImgParams::ImgParams( const int t, const int d, const cv::Size& d_kernel, const cv::Size& e_kernel, algorithm_t a ): threshold(t), dilate(d), dilate_kernel_size(d_kernel), erode_kernel_size(e_kernel), algo_t(a)
{
	dilate_kernel = cv::getStructuringElement( cv::MORPH_CROSS, dilate_kernel_size );
	erode_kernel = cv::getStructuringElement( cv::MORPH_CROSS, erode_kernel_size );

	regions.reserve(32);
	contours.reserve(1028);
	if(algo_t == ALGO_CURRENT)		reccurence = 4;
	if(algo_t == ALGO_CURRENT_GREY)	reccurence = 2;
}

std::mutex mutex_push;
std::vector<tesseract::TessBaseAPI> ocr;

// double optical_flow_to_shift( const cv::Mat& flow, cv::Mat& img )
// {
// 	int step = 8;
// 	double shift_x = 0;
// 	int cnt = 0;

// 	for( int x = 0; x < img.rows; x += step )
// 	{
// 		for( int y = 0; y < img.cols; y += step )
// 		{
// 			cv::Point2f pt = flow.at<cv::Point2f>(x, y);
// 			shift_x += std::abs(pt.x);
// 			++cnt;
// 			// std::cout << shift_x/cnt << std::endl;
// 		}
// 	}
// 	return shift_x/cnt;
// }

// std::pair<double,double> draw_optical_flow( const cv::Mat& flow, cv::Mat& img )
// {
// 	int stepSize = 5;
// 	// cv::Scalar color = cv::Scalar(255, 255, 255);
// 	cv::Scalar color = cv::Scalar(0,0,0);

// 	double shift_x = 0;
// 	double shift_y = 0;
// 	int cnt = 0;
// 	int y = 0, x = 0;

// 	for(y = 0; y < img.rows; y += stepSize)
// 	{
// 		for(x = 0; x < img.cols; x += stepSize)
// 		{
// 			// Circles to indicate the uniform grid of points
// 			// cv::circle( img, cv::Point(x,y), 2, color, -1 );
// 			// Lines to indicate the motion vectors
// 			cv::Point2f pt = flow.at<cv::Point2f>(y, x);
// 			double res = cv::norm( cv::Point(cvRound(x+pt.x), cvRound(y+pt.y)) - cv::Point(x,y) );
// 			if( res )
// 			{
// 				// std::cout << res << std::endl;
// 				cv::circle( img, cv::Point(cvRound(x+pt.x), cvRound(y+pt.y)), 3	, color, -1 );
// 				// cv::line( img, cv::Point(x,y), cv::Point(cvRound(x+pt.x), cvRound(y+pt.y)), color, 1 );
// 			}
// 			// if( cv::norm(  ) )



// 			shift_x += pt.x;
// 			shift_y += pt.y;
// 			++cnt;
// 		}
// 	}
// 	std::cout << shift_x << "," << shift_y << ": " << cnt << "  " << x*y << std::endl;

// 	return std::make_pair(shift_x/cnt, shift_y/cnt);
// 	// std::cout << cnt << "  " << shift_x << " " << shift_x/cnt;
// }

// cv::Mat get_cropped( cv::Mat input, const cv::Rect& box )
// {
// 	cv::Mat cropped;
// 	cv::getRectSubPix( input, box.size(), cv::Point(box.x+box.width/2, box.y+box.height/2), cropped );
// 	// cv::copyMakeBorder( cropped, cropped, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(0) );
// 	return cropped;
// }

// #include <opencv2/video/tracking.hpp>

// void foo( cv::Mat& img1, cv::Mat& img2 )
// {
// 	cv::Mat flow, flow_gray, img_cropped, img_diff, img_bitwise;

// 	cv::calcOpticalFlowFarneback( img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW );
// 	flow_gray = img1.clone();
// 	auto shift = draw_optical_flow( flow, flow_gray );

// 	// double shift_x = optical_flow_to_shift( flow, flow_gray );
// 	// std::cout << "shift = " << shift.first << ", " << shift.second << std::endl;

// 	// auto cropped1 = get_cropped( img1, cv::Rect( 11, 0, img1.cols, img1.rows) );
// 	// auto cropped2 = get_cropped( img2, cv::Rect( 0, 0, img2.cols, img2.rows) );
// 	// // // cv::imshow( "flow", img1 );
// 	// // cv::imshow( "cropped1", cropped1 );
// 	// // cv::imshow( "cropped2", cropped2 );

// 	// cv::absdiff( cropped1, cropped2, img_cropped );
// 	cv::absdiff( img1, img2, img_diff );
// 	// // cv::bitwise_and( cropped2, img_cropped, img_bitwise );
// 	// cv::imshow( "img", img_cropped );
// 	// cv::imshow( "img", img_diff );
// 	// cv::imshow( "img", img_bitwise );

// 	cv::imshow( "flow", flow_gray );
// 	cv::waitKey(1000);

// 	// draw_optical_flow( flow, flow_gray );

// 	// double shift_x = optical_flow_to_shift( flow, flow_gray );
// 	// // double shift_x = 22;

// 	// std::cout << "optical_flow_to_shift() = " << shift_x << std::endl;

// 	// auto cropped1 = get_cropped( tmp1, cv::Rect( 0, 0, img1.cols, img1.rows) );
// 	// auto cropped2 = get_cropped( tmp2, cv::Rect( 18, 0, img2.cols, img2.rows) );
// 	// // cv::imshow( "flow", img1 );
// 	// cv::imshow( "cropped1", cropped1 );
// 	// cv::imshow( "cropped2", cropped2 );

// 	// auto img_cropped_diff = get_diff( cropped1, cropped2 );
// 	// // img_cropped_diff = get_diff( img_diff, img_cropped_diff );
// 	// cv::imshow( "img", img_cropped_diff );
// }


void tesseract_init()
{
	std::cout << "Tesseract initializing..." << std::endl;
	ocr.resize(OCR_MAX);
	for( auto& tess: ocr )
	{
		// tess.Init( NULL, "eng", tesseract::OEM_TESSERACT_CUBE_COMBINED );
		tess.Init( NULL, "eng", tesseract::OEM_TESSERACT_ONLY );
		tess.SetPageSegMode( tesseract::PSM_SINGLE_WORD );
	}
	std::cout << "Success: tesseract_init()" << std::endl;
}

void set_px_average( cv::Mat& img )
{
	std::vector<pixel_t> corners;

	auto mean = cv::mean(img);
	pixel_t px_average{(unsigned char)mean[0], (unsigned char)mean[1], (unsigned char)mean[2]};
	for( int i = 0; i < img.cols; ++i )
	{
		for( int j = 0; j < img.rows; ++j )
		{
			auto& px = img.at<pixel_t>( cv::Point(i,j) );
			px = px_average;
		}
	}
}

void hide_text( cv::Mat& img )
{
	int BLOCK_CNT = 12;
	int block_width = img.cols/BLOCK_CNT;

	if( img.cols > 0 && img.rows > 0 )
	{
		for( int i = 0; i < BLOCK_CNT; ++i )
		{
			int block_x = 0 + block_width*i;
			if( i == BLOCK_CNT-1 ) block_width = img.cols - block_width*i;
			cv::Rect block{ block_x, 0, block_width, img.rows };
			cv::Mat tmp = img(block);
			set_px_average( tmp );
		}
		cv::blur( img, img, cv::Size(17,17) );
	}
}

bool identify_text( const cv::Mat& img, tesseract::TessBaseAPI& ocr_item )
{
	bool ret = false;
	double alpha_digit = 0;

	ocr_item.SetImage( img.data, img.cols, img.rows, 1, img.step );
	std::string str = ocr_item.GetUTF8Text();

	auto new_end = std::remove_if( str.begin(), str.end(), [](char c) { return (c == '\n' || c == ' ' || c <= 0); } );
	if( new_end != str.end() ) str.erase( new_end, str.end() );

	if(str.size() > 10) str.resize(10);
	// std::cout << "\ttext = " << str << std::endl;
	if(str.size() < 5 || str.size() > 11) return ret;
	// if(str.find('8') == std::string::npos && str.find('B') == std::string::npos) return ret;
	if(str[0] == '0' || str[0] == 'O' || str[0] == 'D' || str[0] == '(' || str[0] == '8' || str[0] == 'G' || str[0] == 'U' || str[0] == 'o' || str[1] == '0' || str[1] == 'O' || str[1] == '1' || str[2] == '0' )
	{
		for( auto c: str )
		{
			if( c == 'l' || c == 'i' ) c = '1';
			if( isupper(c) || isdigit(c) ) ++alpha_digit;
		}
		ret = (alpha_digit/str.size() >= 0.6);
		// if(ret)
		// 	std::cout << frame_cnt << " ---> " << str << "  " << alpha_digit << std::endl;
	}
	return ret;
}


auto npos = std::string::npos;
std::string letter_spec = "0OD8B";
std::string letter_1st = "(GUoJI1\\";
std::string letter_2nd = "0O1IQB";
std::string letter_3rd = "0O8I";

bool identify_text2( const cv::Mat& img, tesseract::TessBaseAPI& ocr_item )
{
	bool ret = false;
	double alpha_digit = 0;

	ocr_item.SetImage( img.data, img.cols, img.rows, 1, img.step );
	std::string str = ocr_item.GetUTF8Text();

	auto new_end = std::remove_if( str.begin(), str.end(), [](char c) { return (c == '\n' || c == ' ' || c <= 0 || c == '\''); } );
	if( new_end != str.end() ) str.erase( new_end, str.end() );

	if(str.size() > 10) str.resize(10);
	// std::cout << "\t" << img.cols << "x" << img.rows << "\ttext = " << str << std::endl;
	if(str.size() < 5 || str.size() > 11) return ret;
	// if(str.find('8') == std::string::npos && str.find('B') == std::string::npos) return ret;


	if (letter_spec.find(str[0]) != npos ||
		(letter_1st.find(str[0]) != npos && str.find_first_of(letter_spec, 1) != npos) ||
		(letter_2nd.find(str[1]) != npos && str.find_first_of(letter_spec, 2) != npos) ||
		(letter_3rd.find(str[2]) != npos && str.find_first_of(letter_spec, 3) != npos) ||
		(str.find("DO") != npos || str.find("D0") != npos || str.find("DD") != npos)) {

	// if( (str[0] == '0' || str[0] == 'O' || str[0] == 'D' || str[0] == '(' || str[0] == '8' || str[0] == 'G' || str[0] == 'U' || str[0] == 'o' || str[0] == 'J' || str[0] == 'I' || str[0] == '1' || str[0] == '\\' || str[1] == '0' || str[1] == 'O' || str[1] == '1' || str[1] == 'Q' || str[1] == 'B' || str[2] == '0' || str[2] == '8' || str[2] == 'I') ||
	// 	(str.find("DO") != std::string::npos || str.find("D0") != std::string::npos || str.find("DD") != std::string::npos))
	// {
		for (auto c: str) {
			if (c == 'l' || c == 'i') c = '1';
			if (isupper(c) || isdigit(c)) ++alpha_digit;
		}
		ret = (alpha_digit/str.size() >= 0.6);
		if (ret) {
			std::cout << frame_cnt << " ---> " << img.cols << "x" << img.rows << " > " << str << "  " << alpha_digit << std::endl;
		}
		// else
		// 	std::cout << "\tFake by alpha: " << img.cols << "x" << img.rows << "\ttext = " << str << std::endl;
	}
	// else
	// {
	// 	std::cout << "\tFake: " << img.cols << "x" << img.rows << "\ttext = " << str << std::endl;
	// }
	return ret;
}

void draw_rectangle( cv::Mat& input, const cv::Rect& box )
{
	cv::rectangle( input, box.tl(), box.br(), CV_RGB(255,0,0), 2, 4 );
	// cv::rectangle( input, box.tl(), box.br(), CV_RGB(255,255,255), 2, 4 );
}

void draw_contours( std::vector<std::vector<cv::Point>> contours, cv::Size size )
{
	cv::Mat image( size, CV_8UC3, CV_RGB(0,0,0) );
	cv::drawContours( image, contours, -1, CV_RGB(255,255,255), -1 );
	cv::imshow( "contours", image );
}

inline void get_contours( const cv::Mat& img, std::vector<std::vector<cv::Point>>& contours )
{
	cv::findContours( img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
}

inline void get_threshold_zero( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_threshold, int value )
{
	cv::gpu::threshold( img, img_threshold, value, 255, CV_THRESH_TOZERO );
}

inline void get_threshold_zero( const cv::Mat& img, cv::Mat& img_threshold, int value )
{
	cv::threshold( img, img_threshold, value, 255, CV_THRESH_TOZERO );
}

inline void get_threshold_bin( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_threshold, int value )
{
	cv::gpu::threshold( img, img_threshold, value, 255, CV_THRESH_BINARY );
}

inline void get_threshold_bin( const cv::Mat& img, cv::Mat& img_threshold, int value )
{
	cv::threshold( img, img_threshold, value, 255, CV_THRESH_BINARY );
}

inline void get_diff( const cv::gpu::GpuMat frame1, const cv::gpu::GpuMat frame2, cv::gpu::GpuMat& img_diff )
{
	cv::gpu::absdiff( frame1, frame2, img_diff );
}

inline void get_dilated( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_dilated, int dilation, const cv::Mat& kernel )
{
	cv::gpu::dilate( img, img_dilated, kernel, cv::Point(-1,-1), dilation );
}

inline void get_dilated( const cv::Mat& img, cv::Mat& img_dilated, int dilation, const cv::Mat& kernel )
{
	cv::dilate( img, img_dilated, kernel, cv::Point(-1,-1), dilation );
}

inline void get_eroded( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_eroded, const cv::Mat& kernel )
{
	cv::gpu::erode( img, img_eroded, kernel );
}

inline void get_eroded( const cv::Mat& img, cv::Mat& img_eroded, const cv::Mat& kernel )
{
	cv::erode( img, img_eroded, kernel );
}

void get_bitand_diff( ImgParams& param, GPUVARS* g )
{
	// cv::gpu::threshold( g->frame_grey_prev, param.gpu_blacked_prev, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::threshold( g->frame_grey_curr, param.gpu_blacked_prev, 220, 255, CV_THRESH_TOZERO );
	// cv::gpu::threshold( g->frame_grey_curr, param.gpu_blacked_curr, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::threshold( g->frame_grey_next, param.gpu_blacked_curr, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::absdiff( param.gpu_blacked_prev, param.gpu_blacked_curr, param.gpu_img_diff );
	cv::gpu::bitwise_and( param.gpu_blacked_curr, param.gpu_img_diff, param.gpu_img );
}

cv::Mat img_hsv;
cv::Scalar GREY_HSV_MIN(0, 0, 130);
cv::Scalar GREY_HSV_MAX(180, 80, 198);
// cv::Scalar GREY_HSV_MIN(0, 5, 152);
// cv::Scalar GREY_HSV_MAX(180, 80, 200);
void get_bitand_diff2( ImgParams& param, GPUVARS* g, const cv::Mat& frame_curr )
{
	cv::gpu::threshold( g->frame_grey_next, param.gpu_img, 193, 255, CV_THRESH_TOZERO_INV );
	cv::gpu::threshold(param.gpu_img, param.gpu_img, 110, 255, CV_THRESH_BINARY);
	param.gpu_img.download(param.img);

	cv::cvtColor(frame_curr, img_hsv, CV_BGR2HSV);
	cv::inRange(img_hsv, GREY_HSV_MIN, GREY_HSV_MAX, img_hsv);

	// param.img = img_hsv.clone();
	cv::bitwise_and(param.img, img_hsv, param.img);
}


void get_bitand_diff3( ImgParams& param, GPUVARS* g )
{
	cv::gpu::threshold( g->frame_grey_curr, param.gpu_blacked_prev, param.threshold, 255, CV_THRESH_TOZERO );
	cv::gpu::threshold( g->frame_grey_next, param.gpu_blacked_curr, param.threshold, 255, CV_THRESH_TOZERO );
	cv::gpu::absdiff(param.gpu_blacked_prev, param.gpu_blacked_curr, param.gpu_img);

	cv::gpu::threshold(param.gpu_img, param.gpu_img, 188, 255, CV_THRESH_TOZERO_INV);
	// // cv::gpu::threshold( g->frame_grey_next, param.gpu_img, 205, 255, CV_THRESH_TOZERO_INV );
	cv::gpu::bitwise_and(param.gpu_img, param.gpu_blacked_curr, param.gpu_img);

	// cv::gpu::threshold(param.gpu_img, param.gpu_img, param.threshold, 255, CV_THRESH_BINARY);
	
	param.gpu_img.download(param.img);
}

void set_black( cv::Mat& img )
{
	for( int i = 0; i < img.rows; ++i )
	{
		for( int j = 0; j < img.cols; ++j )
			img.at<uchar>(i,j) = 0;
	}
}

void find_text_regions( const cv::Mat& img, std::vector<cv::Rect>& regions, std::vector<std::vector<cv::Point>>& contours )
{
	// auto t = img.clone();
	get_contours( img, contours );
	// std::cout << "contours.size() = " << contours.size() << std::endl;
	// draw_contours( contours, img.size() );

	for( auto contour: contours )
	{
		auto rotated_box = cv::minAreaRect(contour);
		// if( rotated_box.angle < 15 || rotated_box.angle > 345 )
		{
			auto box = rotated_box.boundingRect();

			// if( box.width >= 100 && box.width <= 400 && box.height >= 5 && box.height <= 46 && box.width/box.height <= 30 )
			// if( box.width >= 110 && box.width <= 290 )
			{
				// std::cout << "box0 = " << box << std::endl;
				// draw_rectangle(t, box);
				// cv::imshow( "label", img(box) );
				// cv::waitKey(500);
				// regions.push_back(box);
			}

			if( (box.width >= 133 || (box.x > 1700 && box.width >= 110)) && box.width <= 233 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
			{
				if( box.height <= 19 ) { box.y -= 2; box.height += 4; }
				if( box.height > 40 ) 	box.height = 30;
				if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
				{
					// std::cout << "box = " << box << std::endl;
					regions.push_back(box);
				}
			}
		}
	}
}

void find_text_regions2( const cv::Mat& img, std::vector<cv::Rect>& regions, std::vector<std::vector<cv::Point>>& contours )
{
	// auto t = img.clone();
	get_contours( img, contours );
	// std::cout << "contours.size() = " << contours.size() << std::endl;
	// draw_contours( contours, img.size() );

	for( auto contour: contours )
	{
		auto rotated_box = cv::minAreaRect(contour);
		// if( rotated_box.angle < 15 || rotated_box.angle > 345 )
		{
			auto box = rotated_box.boundingRect();

			// if (box.y > 1045) continue;

			// if( box.width >= 140 && box.width <= 260 && box.height >= 5 && box.height <= 46 && box.width/box.height <= 30 )
			// if( box.width >= 110 && box.width <= 290 )
			{
				// std::cout << "box0 = " << box << std::endl;
				// draw_rectangle(t, box);
				// cv::imshow( "label", img(box) );
				// cv::waitKey(500);
				// regions.push_back(box);
			}

			// if( box.width >= 167 && box.width <= 215 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 && (box.y < 500 || box.y > 900) )
			if( box.width >= 167 && box.width <= 187 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 && (box.y < 500 || box.y > 900) )
			{
				if( box.height <= 19 ) { box.y -= 2; box.height += 4; }
				if( box.height > 40 )  box.height = 30;
				if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height <= img.rows )
				{
					// std::cout << "box = " << box << std::endl;
					regions.push_back(box);
				}
			}
		}
	}
}

void del_small_areas( const cv::Mat& img )
{
	cv::Mat img_clone = img.clone();
	std::vector<std::vector<cv::Point>> contours;
	get_contours( img_clone, contours );
	int deleted = 0;

	for( auto contour: contours )
	{
		auto box2 = cv::minAreaRect(contour);
		auto box = box2.boundingRect();

		if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
		{
			if( box.height < 12 || ( box.width < 10 && box.height < 10 ) )
			{
				cv::Mat tmp = img(box);
				set_black( tmp );
			}
		}
	}
	return;
}

void del_small_areas2( const cv::Mat& img )
{
	cv::Mat img_clone = img.clone();
	std::vector<std::vector<cv::Point>> contours;
	get_contours( img_clone, contours );
	int deleted = 0;

	for( auto contour: contours )
	{
		auto box2 = cv::minAreaRect(contour);
		auto box = box2.boundingRect();

		if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
		{
			if( box.height < 5 && box.width < 8 )
			{
				cv::Mat tmp = img(box);
				set_black( tmp );
			}
		}
	}
	return;
}

void del_small_areas3( const cv::Mat& img )
{
	cv::Mat img_clone = img.clone();
	std::vector<std::vector<cv::Point>> contours;
	get_contours( img_clone, contours );
	int deleted = 0;

	for( auto contour: contours )
	{
		auto box2 = cv::minAreaRect(contour);
		auto box = box2.boundingRect();

		if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
		{
			if( box.height < 11 || box.width < 5 )
			{
				cv::Mat tmp = img(box);
				set_black( tmp );
			}
		}
	}
	return;
}

bool find_places_by_size( ImgParams& param, GPUVARS* g, const cv::Mat& frame_curr )
{
	if(param.algo_t == ALGO_CURRENT_GREY) {
		get_bitand_diff2(param, g, frame_curr);
		del_small_areas2(param.img);
		param.img_threshold = param.img.clone();
	}
	else if(param.algo_t == ALGO_DIFF_GREY) {
		get_bitand_diff3(param, g);
		del_small_areas3(param.img);
		param.img_threshold = param.img.clone();
	}
	else {
		if( param.algo_t == ALGO_DIFF ) {
			get_bitand_diff( param, g );
		}
		else {
			cv::gpu::threshold( g->frame_grey_next, param.gpu_img, param.threshold, 255, CV_THRESH_TOZERO );
		}
		get_threshold_bin( param.gpu_img, param.gpu_img_threshold, param.threshold );
		param.gpu_img.download(param.img);
		param.gpu_img_threshold.download(param.img_threshold);
		del_small_areas( param.img_threshold );
	}
	get_eroded( param.img_threshold, param.img_eroded, param.erode_kernel );
	get_dilated( param.img_eroded, param.img_dilated, param.dilate, param.dilate_kernel );
	
	// cv::imshow( "img", param.img );
	// cv::imshow( "img_", param.img_dilated );
	// cv::imshow( "img_", param.img_eroded );
	if (param.algo_t == ALGO_DIFF || param.algo_t == ALGO_CURRENT)
		find_text_regions(param.img_dilated, param.regions, param.contours);
	else {
		find_text_regions2(param.img_dilated, param.regions, param.contours);
	}

	return true;
}

void roi_normalize( cv::Rect& roi, int width, int height )
{
	int limit_width = 210;
	int limit_height = 33;

	if( roi.x > 10 ) roi.x -= 10;
	if( roi.y > 8 ) roi.y -= 8;
	if( roi.width < limit_width && (width - roi.x - limit_width) >= 0 ) roi.width = limit_width;
	if( roi.height < limit_height && (height - roi.y - limit_height) >= 0 ) roi.height = limit_height;
}

bool find_places_by_text( ImgParams& param, const cv::Rect& roi, int ocr_idx )
{
	auto img_roi = param.img(roi);
	bool ret = false;

	if(param.algo_t == ALGO_DIFF || param.algo_t == ALGO_CURRENT)
		ret = identify_text(img_roi, ocr[ocr_idx]);
	else
		ret = identify_text2(img_roi, ocr[ocr_idx]);

	if (ret) {
		mutex_push.lock();
		// std::cout << "\t\t" << roi << std::endl;
		if (param.roi_place.area() > 0)
			param.roi_place_old = param.roi_place;
		param.roi_place = roi;
		param.roi_place_origin = roi;
		roi_normalize( param.roi_place, param.img.cols, param.img.rows );
		mutex_push.unlock();
		return true;
	}
	return false;
}

std::vector<std::future<bool>> g_futures;
std::vector<std::thread> g_pool;
std::vector<cv::Rect> g_regions;

bool find_places_entry( std::vector<ImgParams>& params, GPUVARS* g, const cv::Mat& frame_curr )
{
	g_futures.clear();
	g_pool.clear();
	g_regions.clear();

	for( auto& param: params )
	{
		param.regions.erase( param.regions.begin(), param.regions.end() );
		param.contours.erase( param.contours.begin(), param.contours.end() );
		if(param.algo_t == ALGO_CURRENT || param.algo_t == ALGO_CURRENT_GREY)
		{
			if( param.reccurence > 1 && frame_cnt % param.reccurence != 0 ) continue;
		}
		g_futures.push_back( std::async( find_places_by_size, std::ref(param), g, std::cref(frame_curr) ) );
	}
	for( auto& f: g_futures ) f.get();

	int ocr_idx = 0;
	for( auto& param: params )
	{
		for( auto roi: param.regions )
		{
			auto it = std::find_if( g_regions.begin(), g_regions.end(), [=] (cv::Rect r) {
																		auto cross = r & roi;
																		return ((double)cross.area()/r.area() >= 0.9);
																	} );
			if( it == g_regions.end() && ocr_idx < OCR_MAX )
			{
				// cv::imshow( "label", param.img(roi) );
				// cv::waitKey(2000);
				auto cross = param.roi_place_origin & roi;
				// std::cout << (double)cross.area()/roi.area() << std::endl;
				if( (double)cross.area()/roi.area() >= 0.8 ) continue;

				if( param.algo_t == ALGO_CURRENT )
				{
					cv::Mat img_roi = param.img(roi);					
					worker.add( std::make_tuple(roi, img_roi) );
					continue;
				}
				g_pool.push_back( std::thread( find_places_by_text, std::ref(param), roi, ocr_idx ) );
				g_regions.push_back(roi);
				++ocr_idx;
			}
		}
	}
	for( auto& t: g_pool ) t.join();

	return true;
}

double tm_full = 0;
double tm_full_prev = 0;
int stat_size = 100;

void img_detect_label( cv::Mat& frame_curr, std::vector<ImgParams>& params, GPUVARS* g )
{
	++frame_cnt;

	// if( frame_cnt < 900 ) return;
	// cv::imwrite("4.jpg", frame_curr);
	// std::cout << frame_cnt << std::endl;
	if( g )
	{
		auto beg = cv::getTickCount();
		if( !frame_curr.empty() )
		{
			g->frame_curr.upload(frame_curr);
			cv::gpu::cvtColor( g->frame_curr, g->frame_grey_next, CV_BGR2GRAY );

			if( !g->frame_grey_curr.empty() )
			{
				find_places_entry( params, g, frame_curr );

				for (auto param: params) {
					cv::Mat tmp = frame_curr(param.roi_place);
					hide_text(tmp);
					if (param.algo_t == ALGO_CURRENT_GREY) {
						tmp = frame_curr(param.roi_place_old);
						hide_text(tmp);
					}
					// draw_rectangle(frame_curr, param.roi_place);
				}

				for (auto param: params) {
					if (param.algo_t == ALGO_DIFF_GREY || param.algo_t == ALGO_CURRENT_GREY) {
						cv::Rect r{12, 1052, 1900, 26};
						cv::Mat tmp2 = frame_curr(r);
						hide_text(tmp2);
						break;
					}
				}

				std::lock_guard<std::mutex> lg(g_mutex);
				{
					for( auto& r: g_results )
					{
						if( r.left > 0 )
						{
							cv::Mat tmp = frame_curr(r.roi_norm);
							hide_text( tmp );
							// draw_rectangle(frame_curr, r.roi_norm);
							--r.left;
						}
					}
				}
				// cv::imshow("detector", frame_curr);
				// cv::waitKey(500);
			}
			g->frame_prev = g->frame_curr.clone();
		}
		// g->frame_grey_prev = g->frame_grey_curr.clone();
		// g->frame_grey_prev = g->frame_grey_curr.clone();
		g->frame_grey_curr = g->frame_grey_next.clone();

		double frame_tm = ((double)cv::getTickCount() - beg)*1000.0/cv::getTickFrequency();
		tm_full += frame_tm;

		if( frame_cnt % stat_size != 0 );
		else
		{
			std::cout << frame_cnt << ": " << tm_full << " msec, average: " << tm_full/frame_cnt << " msec/frame, delta_average: " << (tm_full - tm_full_prev)/stat_size << " msec/frame" << std::endl;
			tm_full_prev = tm_full;
		}
	}
}
