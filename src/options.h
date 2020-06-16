#pragma once

#include <cstdint>
#include <exception>

class Options {
public:
    Options(int argc, const char* argv[]);
    ~Options() = default;

    // Benchmark options.
    bool benchmark {};
    bool benchmark_next_scenes {};
    uint32_t benchmark_max_time {};

    // Scene options.
    uint32_t scene_index {};
    uint32_t algorithm_index {};

    // Renderer options.
    uint32_t samples {};
    uint32_t bounces {};
    uint32_t max_samples {};

    // Window options
    uint32_t width {};
    uint32_t height {};
    bool fullscreen {};
};
