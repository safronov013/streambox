#pragma once
#include <iostream>
#include <deque>
#include <thread>
#include <algorithm>
#include <future>
#include <memory>
#include <condition_variable>

#include <opencv2/core.hpp>
#include <tesseract/baseapi.h>


typedef std::tuple<cv::Rect,cv::Mat> RoiData;
typedef std::pair<cv::Rect,cv::Rect> RoiResult;

typedef struct _ROI_RESULT_S
{
	cv::Rect roi_real;
	cv::Rect roi_norm;
	int left;
} ROI_RESULT;

extern std::deque<ROI_RESULT> g_results;
extern std::mutex g_mutex;


class Thread
{
	RoiData m_roi;
	std::thread m_handle;
public:
	tesseract::TessBaseAPI m_ocr;
	bool m_status;
	Thread();
	~Thread();
	void tesseract_init();
	void start( RoiData& roi );
	void process_roi();
};

typedef std::unique_ptr<Thread> ThreadPtr;


class QueueManager
{
	const int m_thread_max = 6;
	std::vector<ThreadPtr> m_threads;
	std::deque<RoiData> m_pool;
	std::mutex m_mutex;
	std::condition_variable m_cv;
public:
	QueueManager();
	~QueueManager();
	void listen();
	void add( RoiData&& roi );
	void process();
};
