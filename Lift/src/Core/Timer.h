#pragma once

#include <chrono>

class Timer {
public:
	Timer();
	~Timer() = default;

	static void Start();
	static void Stop();
	static void Reset();
	static void Restart();
	static void Tick();

	static float GetElapsedTime();

	static bool IsRunning() { return is_running_; }

	// seconds passed since last frame
	static float DeltaTime;
private:
	static std::chrono::system_clock::time_point begin_;
	static std::chrono::system_clock::time_point last_time_point_;
	static bool is_running_;
	static float elapsed_time_;
};
