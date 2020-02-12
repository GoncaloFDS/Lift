#pragma once

#include <cstdint>
#include <string>
#include "events/event.h"

namespace vulkan {
struct WindowData final {
    using EventCallBackFn = std::function<void(Event&)>;

    std::string title;
    uint32_t width;
    uint32_t height;
    bool cursorDisabled;
    bool fullscreen;
    bool resizable;

     EventCallBackFn eventCallbackFn;
};
}
