#pragma once

#include <cstdint>
#include <exception>

class Options {
public:
    Options(int argc, const char* argv[]);
    ~Options() = default;

    // Application options.
    bool benchmark {};
    //
    // Benchmark options.
    bool benchmarkNextScenes {};
    uint32_t benchmarkMaxTime {};

    // Scene options.
    uint32_t sceneIndex {};

    // Renderer options.
    uint32_t samples {};
    uint32_t bounces {};
    uint32_t maxSamples {};

    // Window options
    uint32_t width {};
    uint32_t height {};
    bool fullscreen {};
    bool vSync {};
};
