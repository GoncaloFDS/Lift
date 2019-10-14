#include "pch.h"
#include "timer.h"

std::chrono::system_clock::time_point Timer::k_Begin{};
std::chrono::system_clock::time_point Timer::k_LastTimePoint{};
bool Timer::k_IsRunning = false;
float Timer::elapsedTime = 0.0f;
float Timer::deltaTime = 0.0f;

void Timer::start() {
    if (!k_IsRunning) {
        k_IsRunning = true;
        k_Begin = std::chrono::system_clock::now();
        k_LastTimePoint = std::chrono::system_clock::now();
        elapsedTime = 0.0f;
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
    elapsedTime = 0;
}

void Timer::tick() {
    if (k_IsRunning) {
        const auto now = std::chrono::system_clock::now();
        deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - k_LastTimePoint).count() / 1000000.0f;
        k_LastTimePoint = now;
        elapsedTime += deltaTime;
    } else {
        deltaTime = 0.0f;
    }
}

