#pragma once
#include <iostream>
#include <deque>
#include <thread>


class QueueManager
{
	const int thread_max = 4;
	std::deque<std::thread> pool;
public:
	QueueManager();
	~QueueManager();
};
