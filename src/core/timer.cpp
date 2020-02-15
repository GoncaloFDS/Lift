
#include "timer.h"

std::chrono::system_clock::time_point Timer::k_begin_ {};
std::chrono::system_clock::time_point Timer::k_last_time_point_ {};
bool Timer::k_is_running_ = false;
float Timer::elapsedTime = 0.0f;
float Timer::deltaTime = 0.0f;

void Timer::start() {
    if (!k_is_running_) {
        k_is_running_ = true;
        k_begin_ = std::chrono::system_clock::now();
        k_last_time_point_ = std::chrono::system_clock::now();
        elapsedTime = 0.0f;
    }
}

void Timer::stop() {
    k_is_running_ = false;
}

void Timer::reset() {
    k_begin_ = std::chrono::system_clock::now();
}

void Timer::restart() {
    k_is_running_ = true;
    elapsedTime = 0;
}

void Timer::tick() {
    if (k_is_running_) {
        const auto now = std::chrono::system_clock::now();
        deltaTime =
            std::chrono::duration_cast<std::chrono::microseconds>(now - k_last_time_point_).count() / 1000000.0f;
        k_last_time_point_ = now;
        elapsedTime += deltaTime;
    } else {
        deltaTime = 0.0f;
    }
}
