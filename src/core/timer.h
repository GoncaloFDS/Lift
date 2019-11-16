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

    static auto isRunning() -> bool { return k_IsRunning; }

    // seconds passed since last frame
    static float deltaTime;
    static float elapsedTime;
private:
    static std::chrono::system_clock::time_point k_Begin;
    static std::chrono::system_clock::time_point k_LastTimePoint;
    static bool k_IsRunning;
};
