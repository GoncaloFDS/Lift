#include "pch.h"
#include "Timer.h"

std::chrono::system_clock::time_point Timer::begin_{};
std::chrono::system_clock::time_point Timer::last_time_point_{};
bool Timer::is_running_ = false;
float Timer::elapsed_time_ = 0.0f;
float Timer::DeltaTime = 0.0f;

void Timer::Start() {
	if (!is_running_) {
		is_running_ = true;
		begin_ = std::chrono::system_clock::now();
		last_time_point_ = std::chrono::system_clock::now();
		elapsed_time_ = 0.0f;
	}
}

void Timer::Stop() {
	is_running_ = false;
}

void Timer::Reset() {
	begin_ = std::chrono::system_clock::now();
}

void Timer::Restart() {
	is_running_ = true;
	elapsed_time_ = 0;
}

void Timer::Tick() {
	if (is_running_) {
		const auto now = std::chrono::system_clock::now();
		DeltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time_point_).count() / 1000000.0f;
		last_time_point_ = now;
		elapsed_time_ += DeltaTime;
	}
	else {
		DeltaTime = 0.0f;
	}
}

float Timer::GetElapsedTime() {
	return elapsed_time_;
}
