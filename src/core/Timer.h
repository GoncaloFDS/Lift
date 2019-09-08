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

    static float getElapsedTime();

    static bool isRunning() { return k_IsRunning; }

    // seconds passed since last frame
    static float k_DeltaTime;
private:
    static std::chrono::system_clock::time_point k_Begin;
    static std::chrono::system_clock::time_point k_LastTimePoint;
    static bool k_IsRunning;
    static float k_ElapsedTime;
};
