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


// typedef cv::Rect RoiData;
typedef std::tuple<cv::Rect,cv::Mat> RoiData;


class Thread
{
	RoiData m_roi;
	std::thread m_handle;
public:
	tesseract::TessBaseAPI m_ocr;
	bool m_status;
	Thread();
	~Thread();
	void start( RoiData& roi );
	void tesseract_init();
	void foo();
};

typedef std::unique_ptr<Thread> ThreadPtr;
// typedef std::thread ThreadPtr;

class QueueManager
{
	const int m_thread_max = 4;
	std::vector<ThreadPtr> m_threads;
	std::deque<RoiData> m_pool;
	std::deque<RoiData> m_results;
	std::mutex m_mutex;
	std::condition_variable m_cv;
public:
	QueueManager();
	~QueueManager();
	void listen();
	void add( RoiData&& roi );
	void process();
};
