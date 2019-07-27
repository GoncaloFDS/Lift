#pragma once

#include <chrono>

class Profiler {
public:
	Profiler(std::string message = "");
	~Profiler();

private:
	std::chrono::time_point<std::chrono::steady_clock> start_, end_;
	std::chrono::duration<float> duration_;
	std::string message_;

};
