#pragma once

#include <cstdint>
#include <string>

namespace vulkan {
struct WindowProperties final {
    std::string title;
    uint32_t width;
    uint32_t height;
    bool cursorDisabled;
    bool fullscreen;
    bool resizable;
};
}
