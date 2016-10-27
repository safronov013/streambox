#include <iostream>
#include <deque>
#include <thread>
#include <algorithm>
#include "queue_manager.h"
#include "detector.h"



Thread::Thread()
{
	std::cout << "Thread tesseract_init()..." << std::endl;
	tesseract_init();
}

void Thread::foo()
{
	// std::cout << "start: " << std::get<0>(m_roi) << std::endl;
	// std::this_thread::sleep_for( std::chrono::seconds(1) );
	if( identify_text( std::get<1>(m_roi), m_ocr ) )
	{
		std::cout << "Succeeded: " << std::get<0>(m_roi) << std::endl;
	}
	m_status = false;
}

void Thread::start( RoiData& roi )
{
	if( m_handle.joinable() ) m_handle.join();
	m_status = true;
	m_roi = std::move(roi);
	m_handle = std::thread( &Thread::foo, this );
}

void Thread::tesseract_init()
{
	m_ocr.Init( NULL, "eng", tesseract::OEM_TESSERACT_ONLY );
	m_ocr.SetPageSegMode( tesseract::PSM_SINGLE_WORD );
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
		// std::cout << "Free thread founded!" << std::endl;
		auto roi = std::move(m_pool.front());
		m_pool.pop_front();
		(*it)->start(roi);
	}
	else
		std::cout << "No Free thread" << std::endl;
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
