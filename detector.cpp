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
	if( algo_t == ALGO_CURRENT ) reccurence = 1;
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

bool identify_text( const cv::Mat& img, tesseract::TessBaseAPI& ocr_item )
{
	bool ret = false;
	double alpha_digit = 0;

	ocr_item.SetImage( img.data, img.cols, img.rows, 1, img.step );
	std::string str = ocr_item.GetUTF8Text();

	auto new_end = std::remove_if( str.begin(), str.end(), [](char c) { return (c == '\n' || c == ' ' || c <= 0); } );
	if( new_end != str.end() ) str.erase( new_end, str.end() );

	if( str.size() > 10 ) str.resize(10);
	if( str.size() >= 6 && str.size() < 14 && (str[0] == '0' || str[0] == 'O' || str[0] == 'D' || str[0] == '(' || str[0] == '8' || str[0] == 'G' || str[0] == 'U' || str[0] == 'o' || str[1] == '0' || str[1] == 'O' || str[2] == '0' ) )
	{
		for( auto c: str )
		{
			// printf( "%d=%X=%c\n", c, c, c );
			if( c == 'l' || c == 'i' ) c = '1';
			if( isupper(c) || isdigit(c) ) ++alpha_digit;
		}
		ret = (alpha_digit/str.size() >= 0.6);
		if(ret)
		{
			// if( str.size() == 8 && alpha_digit/str.size() == 1 )
			// 	searched_str = str;
			std::cout << frame_cnt << " ---> " << str << "  " << alpha_digit << std::endl;
		}
		else
			std::cout << alpha_digit/str.size() << std::endl;
	}
	std::cout << "\ttext = " << str << std::endl;
	std::cout << "\talpha_digit = " << alpha_digit << std::endl;

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
	// cv::waitKey(2000);

	for( auto contour: contours )
	{
		auto rotated_box = cv::minAreaRect(contour);
		// if( rotated_box.angle < 15 || rotated_box.angle > 345 )
		{
			auto box = rotated_box.boundingRect();

			// if( box.width >= 70 && box.width <= 400 && box.height >= 5 && box.height <= 46 && box.width/box.height <= 30 )
			// if( box.width >= 95 && box.width <= 250 )
			// {
			// 	std::cout << "box0 = " << box << std::endl;
			// 	// cv::imshow( "label", img(box) );
			// 	// cv::waitKey(500);

			// }


			// if( box.width >= 171 && box.width <= 223 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
			if( (box.width >= 171 || (box.x > 1700 && box.width >= 150)) && box.width <= 223 && box.height >= 16 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
			{
				if( box.height <= 19 ) { box.y -= 2; box.height += 4; }
				if( box.height > 40 ) 	box.height = 30;
				if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
				{
					std::cout << "box = " << box << std::endl;
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
		cv::gpu::threshold( g->frame_curr_gray, param.gpu_img, 170, 255, CV_THRESH_TOZERO );
	}
	param.gpu_img.download(param.img);

	del_small_areas( param.img );

	get_threshold_bin( param.gpu_img, param.gpu_img_threshold, param.threshold );
	param.gpu_img_threshold.download(param.img_threshold);

	get_eroded( param.img_threshold, param.img_eroded, param.erode_kernel );
	get_dilated( param.img_eroded, param.img_dilated, param.dilate, param.dilate_kernel );

	// cv::imshow( "img", param.img );
	cv::imshow( "img_", param.img_dilated );
	cv::waitKey(1000);
	find_text_regions( param.img_dilated, param.regions, param.contours );

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

	if( identify_text( img_roi, ocr[ocr_idx] ) )
	{
		mutex_push.lock();
		param.roi_place = roi;
		roi_normalize( param.roi_place, param.img.cols, param.img.rows );
		mutex_push.unlock();
		return true;
	}
	return false;
}

std::vector<std::future<bool>> g_futures;
std::vector<std::thread> g_pool;
std::vector<cv::Rect> g_regions;

bool find_places_entry( std::vector<ImgParams>& params, GPUVARS* g )
{
	g_futures.clear();
	g_pool.clear();
	g_regions.clear();

	for( auto& param: params )
	{
		param.regions.erase( param.regions.begin(), param.regions.end() );
		param.contours.erase( param.contours.begin(), param.contours.end() );
		if( param.algo_t == ALGO_CURRENT )
		{
			if( param.reccurence > 1 && frame_cnt % param.reccurence != 0 ) continue;
		}
		g_futures.push_back( std::async( find_places_by_size, std::ref(param), g ) );
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
	std::cout << "#" << frame_cnt << "-------------------" << std::endl;

	if( frame_cnt < 11 ) return;
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
					// cv::Mat tmp = frame_curr(param.roi_place);
					// hide_text( tmp );
					draw_rectangle(frame_curr, param.roi_place);
				}

				std::lock_guard<std::mutex> lg(g_mutex);
				{
					for( auto& r: g_results )
					{
						if( r.left > 0 )
						{
							// cv::Mat tmp = frame_curr(r.roi_norm);
							// hide_text( tmp );
							draw_rectangle(frame_curr, r.roi_norm);
							--r.left;
						}
					}
				}
				// cv::imshow( "detector", frame_curr );
				// cv::waitKey(500);
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
			// std::cout << frame_cnt << ": " << tm_full << " msec, average: " << tm_full/frame_cnt << " msec/frame, delta_average: " << (tm_full - tm_full_prev)/stat_size << " msec/frame" << std::endl;
			tm_full_prev = tm_full;
		}
	}
}
