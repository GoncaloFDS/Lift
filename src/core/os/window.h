#pragma once

#include "pch.h"

#include "events/event.h"
#include "renderer/graphics_context.h"

namespace lift {

struct WindowProperties {
    std::string title;
    unsigned int width;
    unsigned int height;
    unsigned int x, y;

    explicit WindowProperties(std::string title = "lift Engine",
                              const unsigned int width = 1280,
                              const unsigned int height = 720,
                              const unsigned int position_x = 200,
                              const unsigned int position_y = 200)
        : title(std::move(title)), width(width), height(height), x(position_x), y(position_y) {
    }

};

// Window Interface to be implemented for each platform
class Window {
public:
    using EventCallbackFn = std::function<void(Event&)>;

    virtual ~Window() = default;

    virtual void onUpdate() = 0;

    [[nodiscard]] virtual auto width() const -> unsigned int = 0;
    [[nodiscard]] virtual auto height() const -> unsigned int = 0;
    [[nodiscard]] virtual auto size() const -> ivec2 = 0;
    [[nodiscard]] virtual auto aspectRatio() const -> float = 0;
    [[nodiscard]] virtual auto getPosition() const -> std::pair<int, int> = 0;

    // Window attributes
    virtual void setEventCallback(const EventCallbackFn& callback) = 0;
    virtual void setVSync(bool enabled) = 0;
    [[nodiscard]] virtual auto isVSync() const -> bool = 0;

    [[nodiscard]] virtual auto getNativeWindow() const -> void* = 0;

    static auto create(const WindowProperties& props = WindowProperties()) -> Window*;

};
}
