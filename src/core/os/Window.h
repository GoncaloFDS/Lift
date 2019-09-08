#pragma once

#include "pch.h"

#include "events/Event.h"
#include "Renderer/GraphicsContext.h"

namespace lift {

struct WindowProperties {
    std::string title;
    unsigned int width;
    unsigned int height;
    unsigned int x, y;

    WindowProperties(std::string title = "lift Engine",
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
    using EventCallbackFn = std::function<void(Event &)>;

    virtual ~Window() = default;

    virtual void onUpdate() = 0;

    [[nodiscard]] virtual unsigned int width() const = 0;
    [[nodiscard]] virtual unsigned int height() const = 0;
    [[nodiscard]] virtual std::pair<int, int> getPosition() const = 0;

    // Window attributes
    virtual void setEventCallback(const EventCallbackFn &callback) = 0;
    virtual void setVSync(bool enabled) = 0;
    virtual bool isVSync() const = 0;

    virtual void *getNativeWindow() const = 0;

    static Window *create(const WindowProperties &props = WindowProperties());

};
}
