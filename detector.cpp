#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
#include <iterator>
#include <future>
#include "detector.h"


cv::Mat g_tmp;


ImgParams::ImgParams()
{

}

ImgParams::ImgParams( const int t, const int d, const cv::Size& d_kernel, const cv::Size& e_kernel, algorithm_t a ): threshold(t), dilate(d), dilate_kernel_size(d_kernel), erode_kernel_size(e_kernel), algo_t(a)
{
	dilate_kernel = cv::getStructuringElement( cv::MORPH_CROSS, dilate_kernel_size );
	erode_kernel = cv::getStructuringElement( cv::MORPH_CROSS, erode_kernel_size );

	// regions.reserve(32);
	// contours.reserve(1028);
}

// friend std::ostream& ImgParams::operator<< ( std::ostream& ostr, const ImgParams& param )
// {
// 	ostr << "[" << param.threshold << "," << param.dilate << "] " << param.dilate_kernel << param.erode_kernel << " regions = " << param.regions.size();
// 	return ostr;
// }


places_t places;
std::mutex mutex_push;
int g_frame_pos = 0;
std::string searched_str("");
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

pixel_t get_pixel_average( const cv::Mat& img )
{
	pixel_t px_average{0, 0, 0};

	for( int i = 1; i < 8; ++i )
	{
		for( int j = 1; j < img.rows-1; ++j )
		{
			auto px = img.at<pixel_t>( cv::Point(i,j) );
			if( px_average[0] == 0 && px_average[1] == 0 && px_average[2] == 0 )
			{
				px_average = px;
			}
			else
			{
				px_average[0] = (px_average[0] + px[0])/2;
				px_average[1] = (px_average[1] + px[1])/2;
				px_average[2] = (px_average[2] + px[2])/2;
			}
		}
	}

	for( int i = img.cols-8; i < img.cols-1; ++i )
	{
		for( int j = 1; j < img.rows-1; ++j )
		{
			auto px = img.at<pixel_t>( cv::Point(i,j) );
			if( px_average[0] == 0 && px_average[1] == 0 && px_average[2] == 0 )
			{
				px_average = px;
			}
			else
			{
				px_average[0] = (px_average[0] + px[0])/2;
				px_average[1] = (px_average[1] + px[1])/2;
				px_average[2] = (px_average[2] + px[2])/2;
			}
			// std::cout << "px_average = " << px << px_average << std::endl;
		}
	}
	return px_average;
}

void set_px_average( cv::Mat& img )
{
	std::vector<pixel_t> corners;

	auto px_average = get_pixel_average(img);

	for( int i = 0; i < img.cols; ++i )
	{
		for( int j = 0, k = 0; j < img.rows; ++j, ++k )
		{
			if( k == 4 ) k = 0;
			auto& px = img.at<pixel_t>( cv::Point(i,j) );
			if( px[0] > 30 && px[1] > 30 && px[2] > 30 )
			{
				px = px_average;
			}
		}
	}
}

void hide_text( cv::Mat& img )
{
	int BLOCK_CNT = 8;
	int block_width = img.cols/BLOCK_CNT;

	for( int i = 0; i < BLOCK_CNT; ++i )
	{
		int block_x = 0 + block_width*i;
		if( i == BLOCK_CNT-1 ) block_width = img.cols - block_width*i;
		cv::Rect block{ block_x, 0, block_width, img.rows };
		// std::cout << "\tblock = " << block << std::endl;
		cv::Mat tmp = img(block);
		set_px_average( tmp );
	}
	// cv::imshow( "block", img(roi) );
	// cv::waitKey(1000);

	for( int i = 1; i < 10; i +=2 )
		cv::GaussianBlur( img, img, cv::Size(i,i), 0, 0 );
}

bool identify_text( const cv::Mat& img, int idx )
{
	bool ret = false;

	ocr[idx].SetImage( img.data, img.cols, img.rows, 1, img.step );
	std::string str = ocr[idx].GetUTF8Text();
	double alpha_digit = 0;

	auto new_end = std::remove_if( str.begin(), str.end(), [](char c) { return (c == '\n'); } );
	if( new_end != str.end() )
		str.erase( new_end, str.end() );

	if( str.size() > 10 ) str.resize(10);
	// if( str.size() > 6 && str.size() < 12 && (str[0] == '0' || str[0] == 'O' || str[0] == '8' || str[0] == 'J' || str[0] == 'G' || str[0] == 'o' || str[1] == '0' ) )
	if( str.size() > 6 && str.size() < 14 && (str[0] == '0' || str[0] == 'O' || str[0] == 'D' || str[0] == '(' || str[0] == '8' || str[0] == 'G' || str[0] == 'o' || str[1] == '0' || str[1] == 'O' || str[2] == '0' ) )
	{
		for( auto c: str )
		{
			if( c == 'l' || c == 'i' ) c = '1';
			if( isupper(c) || isdigit(c) ) ++alpha_digit;
		}
		// std::cout << "alpha_digit/str.size() = " << alpha_digit/str.size() << std::endl;
		ret = (alpha_digit/str.size() >= 0.6);
		if(ret)
		{
			// if( str.size() == 8 && alpha_digit/str.size() == 1 )
			// 	searched_str = str;
			std::cout << str << "  " << alpha_digit << std::endl;
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

bool is_founded( const cv::Rect& box )
{
	for( auto place: places )
	{
		auto roi_intersect = place & box;
		if( (double)roi_intersect.area() / place.area() >= 0.9 )
			return true;
	}
	return false;
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
	// std::vector<std::vector<cv::Point>> contours;
	cv::findContours( img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
	// return contours;
}

// cv::Mat get_threshold_zero( const cv::Mat img, int value )
// cv::gpu::GpuMat get_threshold_zero( const cv::gpu::GpuMat img, cv::gpu::GpuMat& img_threshold, int value )
inline void get_threshold_zero( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_threshold, int value )
{
	// cv::Mat img_threshold;
	cv::gpu::threshold( img, img_threshold, value, 255, CV_THRESH_TOZERO );
	// return img_threshold;
}

inline void get_threshold_zero( const cv::Mat& img, cv::Mat& img_threshold, int value )
{
	cv::threshold( img, img_threshold, value, 255, CV_THRESH_TOZERO );
}

// cv::Mat get_threshold_bin( const cv::Mat img, int value )
inline void get_threshold_bin( const cv::gpu::GpuMat img, cv::gpu::GpuMat& img_threshold, int value )
{
	// cv::Mat img_threshold;
	cv::gpu::threshold( img, img_threshold, value, 255, CV_THRESH_BINARY );
	// return img_threshold;
}

inline void get_threshold_bin( const cv::Mat& img, cv::Mat& img_threshold, int value )
{
	cv::threshold( img, img_threshold, value, 255, CV_THRESH_BINARY );
}

// cv::Mat get_diff( const cv::Mat frame1, const cv::Mat frame2 )
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

// cv::Mat get_eroded( const cv::Mat& img, const cv::Size& kernel_size )
inline void get_eroded( const cv::gpu::GpuMat& img, cv::gpu::GpuMat& img_eroded, const cv::Mat& kernel )
{
	cv::gpu::erode( img, img_eroded, kernel );
}

inline void get_eroded( const cv::Mat& img, cv::Mat& img_eroded, const cv::Mat& kernel )
{
	cv::erode( img, img_eroded, kernel );
}


cv::Mat get_cropped( cv::Mat input, const cv::Rect& box )
{
	cv::Mat cropped;
	cv::getRectSubPix( input, box.size(), cv::Point(box.x+box.width/2, box.y+box.height/2), cropped );
	return cropped;
}

// cv::Mat get_bitand_diff( const cv::Mat img1, const cv::Mat img2 )
// cv::gpu::GpuMat get_bitand_diff( ImgParams& param, GPUVARS* g )
void get_bitand_diff( ImgParams& param, GPUVARS* g )
{
	cv::gpu::threshold( g->frame_prev_gray, param.gpu_blacked_prev, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::threshold( g->frame_curr_gray, param.gpu_blacked_curr, 220, 255, CV_THRESH_TOZERO );
	cv::gpu::absdiff( param.gpu_blacked_prev, param.gpu_blacked_curr, param.gpu_img_diff );
	// cv::gpu::bitwise_and( param.gpu_blacked_curr, param.gpu_img_diff, param.gpu_img_bitand );
	cv::gpu::bitwise_and( param.gpu_blacked_curr, param.gpu_img_diff, param.gpu_img );
	// return param.gpu_img_bitand;
}

void set_black( cv::Mat& img )
{
	for( int i = 0; i < img.rows; ++i )
	{
		for( int j = 0; j < img.cols; ++j )
			img.at<uchar>(i,j) = 0;
	}
}

// void find_text_regions( const cv::Mat& img, std::vector<cv::Rect>& regions, std::vector<std::vector<cv::Point>>& contours )
std::vector<cv::Rect> find_text_regions( const cv::Mat& img )
{
	std::vector<cv::Rect> regions;
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat img_clone = img.clone();

	get_contours( img_clone, contours );
	std::cout << "contours.size() = " << contours.size() << std::endl;
	// draw_contours( contours, img.size() );

	for( auto contour: contours )
	{
		auto rotated_box = cv::minAreaRect(contour);
		if( rotated_box.angle < 15 || rotated_box.angle > 345 )
		{
			auto box = rotated_box.boundingRect();

			// if( box.width >= 171 && box.width <= 223 && box.height >= 17 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
			// {
			// 	// std::cout << "box = " << box << std::endl;
			// }
			if( box.width >= 171 && box.width <= 223 && box.height >= 17 && box.height <= 46 && box.width/box.height >= 4 && box.width/box.height <= 12 )
			{
				// if( box.width > 220 ) box.width = 210;
				if( box.height > 40 ) box.height = 30;
				if( box.x > 0 && box.y > 0 && box.width > 0 && box.height > 0 && box.x+box.width < img.cols && box.y+box.height < img.rows )
				{
					std::cout << "box = " << box << std::endl;
					regions.push_back(box);
				}
			}
		}
	}
	return regions;
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
				// set_black( img(box) );
			}
		}
	}
	return;
}

// bool is_identical( cv::Rect r1, cv::Rect r2 )
// {
// 	auto roi_intersect = r & roi;
// 	return ((double)roi_intersect.area() / roi.area() >= 0.9 );
// }

// bool find_places_by_size( ImgParams& param, const cv::Mat frame_prev_gray, const cv::Mat frame_curr_gray )
bool find_places_by_size( ImgParams& param, GPUVARS* g )
{
	// param.img = (param.algo_t == ALGO_DIFF) ? get_bitand_diff( g->gpu_frame_prev_gray, g->gpu_frame_curr_gray ) : g->gpu_frame_curr_gray.clone();
	if( param.algo_t == ALGO_DIFF )
	{
		get_bitand_diff( param, g );
		param.gpu_img.download(param.img);
	}
	else
	{
		// param.gpu_img = g->frame_curr_gray.clone();
		g->frame_curr_gray.download(param.img);
	}

	del_small_areas( param.img );

	// get_threshold_bin( param.img, param.img_threshold, param.threshold );
	// get_threshold_bin( param.gpu_img, param.gpu_img_threshold, param.threshold );
	cv::gpu::threshold( param.gpu_img, param.gpu_img_threshold, param.threshold, 255, CV_THRESH_BINARY );
	param.gpu_img_threshold.download(param.img_threshold);

	// cv::Mat img_eroded;
	// get_eroded( param.img_threshold, param.img_eroded, param.erode_kernel );
	cv::erode( param.img_threshold, param.img_eroded, param.erode_kernel );
	// get_eroded( param.gpu_img_threshold, param.gpu_img_eroded, param.erode_kernel );
	// param.gpu_img_eroded.download(param.img_eroded);

	// cv::dilate( param.img_eroded, param.img_dilated, param.dilate_kernel, cv::Point(-1,-1), param.dilate );
	get_dilated( param.img_eroded, param.img_dilated, param.dilate, param.dilate_kernel );
	// get_dilated( param.gpu_img_eroded, param.gpu_img_dilated, param.dilate, param.dilate_kernel );
	// param.gpu_img_dilated.download(param.img_dilated);
	// return false;

	// param.regions = std::move(find_text_regions( param.img_dilated ));
	// param.regions = std::move(find_text_regions( param.img_dilated ));

	// param.regions.erase( param.regions.begin(), param.regions.end() );
	// param.contours.erase( param.contours.begin(), param.contours.end() );

	// cv::imshow( "threshold", param.img_threshold );
	// cv::imshow( "dilated", param.img_dilated );
	// cv::waitKey(100);

	// find_text_regions( param.img_dilated, param.regions, param.contours );
	param.regions = std::move( find_text_regions(param.img_dilated) );
	return true;
}

// places_t find_places_by_text( const cv::Mat img, const cv::Mat img_threshold, const cv::Rect roi )
bool find_places_by_text( const cv::Mat& img, const cv::Mat& img_threshold, const cv::Rect& roi, int ocr_idx )
{
	if( !is_founded( roi ) )
	{
		auto img_roi = img(roi);
		// auto img_roi = get_threshold_bin( img(roi), 20 );
		auto img_roi_threshold = img_threshold(roi);

		// if( identify_text( img_roi ) || identify_text( img_roi_threshold ) )
		// cv::imshow( "img_roi", img_roi );
		// cv::imshow( "img_roi_threshold", img_roi_threshold );
		// cv::waitKey(500);
		// if( identify_text( img_roi_threshold, ocr_idx ) )
		// if( identify_text( img_roi, ocr_idx ) )
		if( identify_text( img_roi, ocr_idx ) )
		{
			mutex_push.lock();
			places.push_back(roi);
			mutex_push.unlock();
			std::cout << g_frame_pos << std::endl;
			return true;
		}
		/*else if( roi.height > 32 )
		{
			roi.height -=12;
			roi.y += 12;
			if( identify_text(img_threshold(roi)) )
				places[g_frame_pos] = roi;
		}
		else if( roi.width > 220 )
		{
			roi.width -=20;
			roi.x += 20;
			if( identify_text(img_bitand(roi)) )
				places[g_frame_pos] = roi;
		}*/
	}
	return false;
}

// bool find_places_entry( cv::Mat& frame_prev_gray, cv::Mat& frame_curr_gray )
bool find_places_entry( std::vector<ImgParams>& params, GPUVARS* g )
{
	std::vector<std::future<bool>> futures;

	for( auto& param: params )
	{
		// futures.push_back( std::async( find_places_by_size, std::ref(param), gpu_frame_prev_gray, gpu_frame_curr_gray ) );
		futures.push_back( std::async( find_places_by_size, std::ref(param), g ) );
	}

	for( auto f = futures.begin(); f != futures.end(); ++f )
	{
		f->get();
	}
	futures.clear();

	std::vector<cv::Rect> regions;
	int ocr_idx = 0;
	for( auto param: params )
	{
		// std::cout << "param.regions.size() = " << param.regions.size() << std::endl;

		for( auto roi: param.regions )
		{
			auto it = std::find_if( regions.begin(), regions.end(), [=] (cv::Rect r) {
																		auto cross = r & roi; return ((double)cross.area()/r.area() >= 0.9);
																	} );
			if( it == regions.end() && ocr_idx < OCR_MAX )
			{
				futures.push_back( std::async( find_places_by_text, param.img, param.img_threshold, roi, ocr_idx ) );
				++ocr_idx;
				regions.push_back(roi);
				// break;
			}
		}
	}

	// if( regions.size() > 1 )
	// 	std::cout << regions.size() << std::endl;

	for( auto f = futures.begin(); f != futures.end(); ++f )
	{
		bool is_found = f->get();
		// if( is_found )
		// 	std::cout << "is_found" << std::endl;
	}
	return true;
}

// cv::Mat img_prev_gray, img_curr_gray;

// void img_detect_label( cv::Mat& frame_curr )

double tm_full = 0;
int frame_cnt = 0;

void img_detect_label( cv::Mat& frame_curr, std::vector<ImgParams>& params, GPUVARS* g )
{
	// g_tmp = frame_curr;
	++frame_cnt;

	if( frame_cnt < 200 ) return;
	auto beg = cv::getTickCount();
	// cv::Mat frame_prev_gray, frame_curr_gray;
	if( g )
	{
		if( !frame_curr.empty() )
		{
			g->frame_curr.upload(frame_curr);
			// cv::cvtColor( frame_curr, img_curr_gray, CV_BGR2GRAY );
			cv::gpu::cvtColor( g->frame_curr, g->frame_curr_gray, CV_BGR2GRAY );
			if( !g->frame_prev_gray.empty() )
			{
				// find_places_entry( g->gpu_frame_prev_gray, g->gpu_frame_curr_gray );
				find_places_entry( params, g );

				if( places.size() )
				{
					draw_rectangle( frame_curr, *places.rbegin() );
				}
				// draw_rectangle( frame_curr, cv::Rect( 10, 10, 50, 50 ) );
			// 	// cv::imshow( "detector", frame_curr );
			// 	// if( cv::waitKey(1) == 27 )
			// 	// 	;
			}
		}
		// img_prev_gray = img_curr_gray.clone();
		g->frame_prev_gray = g->frame_curr_gray.clone();

		double frame_tm = ((double)cv::getTickCount() - beg)*1000.0/cv::getTickFrequency();
		tm_full += frame_tm;
		// if( frame_tm > 40 )
		// 	std::cout << " \tframe_tm = " << frame_tm << std::endl;

		if( frame_cnt % 50 != 0 );
		else
		{
			std::cout << frame_cnt << "   " << tm_full << std::endl;
		}

		// if( frame_cnt != 100 );
		// else
		// {
		// 	std::cout << ++frame_cnt << " " << tm_full << std::endl;
		// 	exit(1);
		// }
	}
	// cv::destroyAllWindows();
}

void img_draw_rect( cv::Mat& frame_curr )
{
	auto beg = cv::getTickCount();
	// cv::Mat frame_prev_gray, frame_curr_gray;

	if( !frame_curr.empty() )
	{
		draw_rectangle( frame_curr, cv::Rect( 100, 100, 200, 30 ) );
	}
	return;
}

