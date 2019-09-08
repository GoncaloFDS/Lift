#include "pch.h"
#include "timer.h"

std::chrono::system_clock::time_point Timer::k_Begin{};
std::chrono::system_clock::time_point Timer::k_LastTimePoint{};
bool Timer::k_IsRunning = false;
float Timer::k_ElapsedTime = 0.0f;
float Timer::k_DeltaTime = 0.0f;

void Timer::start() {
    if (!k_IsRunning) {
        k_IsRunning = true;
        k_Begin = std::chrono::system_clock::now();
        k_LastTimePoint = std::chrono::system_clock::now();
        k_ElapsedTime = 0.0f;
    }
}

void Timer::stop() {
    k_IsRunning = false;
}

void Timer::reset() {
    k_Begin = std::chrono::system_clock::now();
}

void Timer::restart() {
    k_IsRunning = true;
    k_ElapsedTime = 0;
}

void Timer::tick() {
    if (k_IsRunning) {
        const auto now = std::chrono::system_clock::now();
        k_DeltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - k_LastTimePoint).count() / 1000000.0f;
        k_LastTimePoint = now;
        k_ElapsedTime += k_DeltaTime;
    } else {
        k_DeltaTime = 0.0f;
    }
}

float Timer::getElapsedTime() {
    return k_ElapsedTime;
}
