#pragma once

#include <chrono>

class Timer {
public:
    Timer();
    ~Timer() = default;

    static void start();
    static void stop();
    static void restart();
    static void tick();

    static auto isRunning() -> bool { return is_running_; }

    // seconds passed since last frame
    static float delta_time;
    static float elapsed_time;

private:
    static std::chrono::system_clock::time_point last_time_point_;
    static bool is_running_;
};
