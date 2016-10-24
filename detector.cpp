#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
#include <iterator>
#include <future>
#include "detector.h"


int frame_cnt = 0;


ImgParams::ImgParams()
{

}

ImgParams::ImgParams( const int t, const int d, const cv::Size& d_kernel, const cv::Size& e_kernel, algorithm_t a ): threshold(t), dilate(d), dilate_kernel_size(d_kernel), erode_kernel_size(e_kernel), algo_t(a)
{
	dilate_kernel = cv::getStructuringElement( cv::MORPH_CROSS, dilate_kernel_size );
	erode_kernel = cv::getStructuringElement( cv::MORPH_CROSS, erode_kernel_size );

	regions.reserve(32);
	contours.reserve(1028);
	if( algo_t == ALGO_CURRENT ) reccurence = 4;
}

std::mutex mutex_push;
std::vector<tesseract::TessBaseAPI> ocr;

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
		// cv::GaussianBlur( img, img, cv::Size(13,13), 0, 0 );
		cv::blur( img, img, cv::Size(17,17) );
		// for( int i = 1; i < 10; i +=2 )
		// 	cv::GaussianBlur( img, img, cv::Size(i,i), 0, 0 );
	}
}

std::string foo( const cv::Mat& img, int idx )
{
	ocr[idx].SetImage( img.data, img.cols, img.rows, 1, img.step );
	return ocr[idx].GetUTF8Text();
}

void bar( const cv::Mat& img, int idx, std::string& str )
{
	ocr[idx].SetImage( img.data, img.cols, img.rows, 1, img.step );
	str = ocr[idx].GetUTF8Text();
}


// auto d = std::chrono::milliseconds(45);

bool identify_text( const cv::Mat& img, int idx )
{
	bool ret = false;

	double beg = (double)cv::getTickCount();

	// std::string str;
	// std::thread t(foo, std::cref(img), idx, std::ref(str));
	// auto f = std::async( std::launch::async, bar, std::cref(img), idx, std::ref(str) );
	// if( f.wait_for(d) == std::future_status::timeout )
	// {
	// 	std::cout << "kill" << std::endl;
	// 	return false;
	// }


	// auto f = std::async( std::launch::async, foo, std::cref(img), idx );
	// if( f.wait_for(std::chrono::milliseconds(60)) == std::future_status::timeout )
	// {
	// 	// std::cout << "kill" << std::endl;
	// 	return false;
	// }
	// double frame_tm1 = ((double)cv::getTickCount() - beg)*1000.0/cv::getTickFrequency();
	// std::cout << "identify_tm0 = " << frame_tm1 << std::endl;

	// auto f = std::async( foo, std::cref(img), idx );
	// std::string str = f.get();


	ocr[idx].SetImage( img.data, img.cols, img.rows, 1, img.step );
	std::string str = ocr[idx].GetUTF8Text();
	double alpha_digit = 0;

	// double frame_tm = ((double)cv::getTickCount() - beg)*1000.0/cv::getTickFrequency();
	// std::cout << "identify_tm = " << frame_tm << std::endl;

	auto new_end = std::remove_if( str.begin(), str.end(), [](char c) { return (c == '\n'); } );
	if( new_end != str.end() )
		str.erase( new_end, str.end() );

	if( str.size() > 10 ) str.resize(10);
	// if( str.size() > 6 && str.size() < 12 && (str[0] == '0' || str[0] == 'O' || str[0] == '8' || str[0] == 'J' || str[0] == 'G' || str[0] == 'o' || str[1] == '0' ) )
	if( str.size() > 6 && str.size() < 14 && (str[0] == '0' || str[0] == 'O' || str[0] == 'D' || str[0] == '(' || str[0] == '8' || str[0] == 'G' || str[0] == 'U' || str[0] == 'o' || str[1] == '0' || str[1] == 'O' || str[2] == '0' ) )
	{
		for( auto c: str )
		{
			if( c == 'l' || c == 'i' ) c = '1';
			if( isupper(c) || isdigit(c) ) ++alpha_digit;
		}
		ret = (alpha_digit/str.size() >= 0.6);
		// if(ret)
		{
			// if( str.size() == 8 && alpha_digit/str.size() == 1 )
			// 	searched_str = str;
			// std::cout << frame_cnt << " ---> " << str << "  " << alpha_digit << std::endl;
		}
	}
	// if( str.size() == 0 )
	{
		// ret = true;
		// std::cout << "\ttext = " << str << std::endl;
		// std::cout << "\talpha_digit = " << alpha_digit << std::endl;
	}
	return ret;
}

void draw_rectangle( cv::Mat& input, const cv::Rect& box )
{
	cv::rectangle( input, box.tl(), box.br(), CV_RGB(255,0,0), 2, 4 );
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
	cv::gpu::threshold( g->frame_prev_gray, param.gpu_blacked_prev, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::threshold( g->frame_curr_gray, param.gpu_blacked_curr, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::absdiff( param.gpu_blacked_prev, param.gpu_blacked_curr, param.gpu_img_diff );
	cv::gpu::bitwise_and( param.gpu_blacked_curr, param.gpu_img_diff, param.gpu_img );
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
	get_contours( img, contours );
	// std::cout << "contours.size() = " << contours.size() << std::endl;
	// draw_contours( contours, img.size() );
	// cv::waitKey(500);

	for( auto contour: contours )
	{
		auto rotated_box = cv::minAreaRect(contour);
		if( rotated_box.angle < 15 || rotated_box.angle > 345 )
		{
			auto box = rotated_box.boundingRect();

			if( box.width >= 171 && box.width <= 223 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
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

bool find_places_by_size( ImgParams& param, GPUVARS* g )
{
	if( param.algo_t == ALGO_DIFF )
	{
		get_bitand_diff( param, g );
	}
	else
	{
		// param.gpu_img = g->frame_curr_gray.clone();
		cv::gpu::threshold( g->frame_curr_gray, param.gpu_img, 170, 255, CV_THRESH_TOZERO );
	}
	param.gpu_img.download(param.img);

	del_small_areas( param.img );

	get_threshold_bin( param.gpu_img, param.gpu_img_threshold, param.threshold );
	param.gpu_img_threshold.download(param.img_threshold);

	get_eroded( param.img_threshold, param.img_eroded, param.erode_kernel );
	get_dilated( param.img_eroded, param.img_dilated, param.dilate, param.dilate_kernel );

	// cv::imshow( "img", param.img_dilated );
	// cv::waitKey(1);
	find_text_regions( param.img_dilated, param.regions, param.contours );

	return true;
}

bool find_places_by_text( ImgParams& param, const cv::Rect& roi, int ocr_idx )
{
	auto img_roi = param.img(roi);

	if( identify_text( img_roi, ocr_idx ) )
	{
		if( param.algo_t == ALGO_CURRENT ) param.missed = 0;

		mutex_push.lock();
		param.roi_place = roi;
		if( param.roi_place.x > 10 ) param.roi_place.x -= 10;
		if( param.roi_place.y > 8 ) param.roi_place.y -= 8;
		if( param.roi_place.width < 210 && (param.img.cols - param.roi_place.x - 210) >= 0 ) param.roi_place.width = 210;
		if( param.roi_place.height < 33 && (param.img.rows - param.roi_place.y - 33) >= 0 ) param.roi_place.height = 33;
		mutex_push.unlock();
		return true;
	}
	return false;
}

bool find_places_entry( std::vector<ImgParams>& params, GPUVARS* g )
{
	std::vector<std::future<bool>> futures;

	for( auto& param: params )
	{
		param.regions.erase( param.regions.begin(), param.regions.end() );
		param.contours.erase( param.contours.begin(), param.contours.end() );
		if( param.algo_t == ALGO_CURRENT )
		{
			if( ++param.missed < 10 ) continue;
			if( param.reccurence > 1 && frame_cnt % param.reccurence != 0 ) continue;
		}
		futures.push_back( std::async( find_places_by_size, std::ref(param), g ) );
	}

	for( auto f = futures.begin(); f != futures.end(); ++f )
	{
		f->get();
	}
	futures.clear();

	// std::vector<std::thread> th;

	std::vector<cv::Rect> regions;
	int ocr_idx = 0;
	for( auto& param: params )
	{
		for( auto roi: param.regions )
		{
			auto it = std::find_if( regions.begin(), regions.end(), [=] (cv::Rect r) {
																		auto cross = r & roi;
																		return ((double)cross.area()/r.area() >= 0.9);
																	} );
			if( it == regions.end() && ocr_idx < OCR_MAX )
			{
				futures.push_back( std::async( find_places_by_text, std::ref(param), roi, ocr_idx ) );
				// th.push_back( std::thread( find_places_by_text, std::ref(param), roi, ocr_idx ) );
				++ocr_idx;
				regions.push_back(roi);
			}
		}
	}

	for( auto f = futures.begin(); f != futures.end(); ++f )
		f->get();
	return true;
}

double tm_full = 0;
double tm_full_prev = 0;
int stat_size = 100;

void img_detect_label( cv::Mat& frame_curr, std::vector<ImgParams>& params, GPUVARS* g )
{
	++frame_cnt;

	// if( frame_cnt < 480 ) return;
	if( g )
	{
		auto beg = cv::getTickCount();
		if( !frame_curr.empty() )
		{
			g->frame_curr.upload(frame_curr);
			cv::gpu::cvtColor( g->frame_curr, g->frame_curr_gray, CV_BGR2GRAY );
			if( !g->frame_prev_gray.empty() )
			{
				find_places_entry( params, g );

				for( auto param: params )
				{
					// if( param.places.size() )
					{
						// draw_rectangle( frame_curr, *param.places.rbegin() );
						// cv::Mat tmp = frame_curr(*param.places.rbegin());
						cv::Mat tmp = frame_curr(param.roi_place);
						hide_text( tmp );
					}
				}
				// cv::imshow( "detector", frame_curr );
				// cv::waitKey(1);
			}
		}
		g->frame_prev_gray = g->frame_curr_gray.clone();

		double frame_tm = ((double)cv::getTickCount() - beg)*1000.0/cv::getTickFrequency();
		tm_full += frame_tm;
		// if( frame_tm > 40 )
			// std::cout << "\tframe_cnt = " << frame_cnt << "\tframe_tm = " << frame_tm << std::endl;

		if( frame_cnt % stat_size != 0 );
		else
		{
			// std::cout << frame_cnt << "   " << tm_full/1000 << std::endl;
			std::cout << std::endl;
			std::cout << frame_cnt << ": " << tm_full << " msec, average: " << tm_full/frame_cnt << " msec/frame, delta_average: " << (tm_full - tm_full_prev)/stat_size << " msec/frame" << std::endl;
			tm_full_prev = tm_full;
		}
	}
}
