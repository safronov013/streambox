#include <iostream>
#include <deque>
#include <thread>
#include <algorithm>
#include "queue_manager.h"
#include "detector.h"


Thread::Thread()
{
	tesseract_init();
}

void Thread::tesseract_init()
{
	m_ocr.Init( NULL, "eng", tesseract::OEM_TESSERACT_ONLY );
	m_ocr.SetPageSegMode( tesseract::PSM_SINGLE_WORD );
}

void Thread::start( RoiData& roi )
{
	if( m_handle.joinable() ) m_handle.join();
	m_status = true;
	m_roi = std::move(roi);
	m_handle = std::thread( &Thread::process_roi, this );
}

void Thread::process_roi()
{
	if( identify_text( std::get<1>(m_roi), m_ocr ) )
	{
		std::lock_guard<std::mutex> lg(g_mutex);
		auto it = std::find_if( g_results.begin(), g_results.end(), [=] (const ROI_RESULT& r) {
																		auto cross = r.roi_real & std::get<0>(m_roi);
																		return ((double)cross.area()/r.roi_real.area() >= 0.8);
																	} );

		if( it == g_results.end() )
		{
			cv::Rect roi_fixed = std::get<0>(m_roi);
			roi_normalize( roi_fixed, 1920, 1080 );
			g_results.push_back({std::get<0>(m_roi), roi_fixed, 250});
		}
		while( g_results.size() > 2 ) g_results.pop_front();
	}
	m_status = false;
}


Thread::~Thread()
{
}


QueueManager::QueueManager()
{
	for( int i = 0; i < m_thread_max; ++i )
		m_threads.push_back( ThreadPtr(new Thread()));
}

void QueueManager::listen()
{
	while(true)
	{
		{
			std::unique_lock<std::mutex> ul(m_mutex);
			m_cv.wait( ul, [=]{ return !m_pool.empty(); } );
			process();
		}
	}
}

void QueueManager::process()
{
	auto it = std::find_if( m_threads.begin(), m_threads.end(), [](const ThreadPtr& ptr) { return !ptr->m_status; } );
	if( it != m_threads.end() )
	{
		auto roi = std::move(m_pool.front());
		m_pool.pop_front();
		(*it)->start(roi);
	}
}

void QueueManager::add( RoiData&& roi )
{
	{
		std::lock_guard<std::mutex> lg(m_mutex);
		m_pool.push_back(roi);
	}
	m_cv.notify_one();
}

QueueManager::~QueueManager()
{
}
