
#include "timer.h"

std::chrono::system_clock::time_point Timer::last_time_point_ {};
bool Timer::is_running_ = false;
float Timer::elapsed_time = 0.0f;
float Timer::delta_time = 0.0f;

void Timer::start() {
    if (!is_running_) {
        is_running_ = true;
        last_time_point_ = std::chrono::system_clock::now();
        elapsed_time = 0.0f;
    }
}

void Timer::stop() {
    is_running_ = false;
}

void Timer::restart() {
    is_running_ = true;
    elapsed_time = 0.0f;
}

void Timer::tick() {
    if (is_running_) {
        const auto now = std::chrono::system_clock::now();
        delta_time =
            std::chrono::duration_cast<std::chrono::microseconds>(now - last_time_point_).count() / 1000000.0f;
        last_time_point_ = now;
        elapsed_time += delta_time;
    } else {
        delta_time = 0.0f;
    }
}
