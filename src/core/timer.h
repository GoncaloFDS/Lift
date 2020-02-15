#pragma once

#include <chrono>

class Timer {
public:
    Timer();
    ~Timer() = default;

    static void start();
    static void stop();
    static void reset();
    static void restart();
    static void tick();

    static auto isRunning() -> bool { return k_is_running_; }

    // seconds passed since last frame
    static float deltaTime;
    static float elapsedTime;

private:
    static std::chrono::system_clock::time_point k_begin_;
    static std::chrono::system_clock::time_point k_last_time_point_;
    static bool k_is_running_;
};
