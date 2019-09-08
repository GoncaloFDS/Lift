#pragma once

#include "Event.h"

namespace lift {
class WindowResizeEvent : public Event {
public:
    WindowResizeEvent(unsigned int width, unsigned int height)
        : width_(width), height_(height) {
    }

    [[nodiscard]] unsigned int getWidth() const { return width_; }
    [[nodiscard]] unsigned int getHeight() const { return height_; }

    [[nodiscard]]  std::string toString() const override {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << width_ << " x " << height_;
        return ss.str();
    }

    [[nodiscard]] static EventType getStaticType() { return EventType::WINDOW_RESIZE; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "WindowResize"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }

private:
    unsigned int width_, height_;
};

class WindowCloseEvent : public Event {
public:
    WindowCloseEvent() = default;

    [[nodiscard]] static EventType getStaticType() { return EventType::WINDOW_CLOSE; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "WindowClose"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }
};

class WindowMinimizeEvent : public Event {
public:
    explicit WindowMinimizeEvent(bool is_minimized)
        : is_minimized_(is_minimized) {
    }

    [[nodiscard]] bool getIsMinimized() const { return is_minimized_; }

    [[nodiscard]] static EventType getStaticType() { return EventType::WINDOW_MINIMIZE; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "WindowMinimize"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }
private:
    bool is_minimized_;
};

class AppTickEvent : public Event {
public:
    AppTickEvent() = default;

    [[nodiscard]] static EventType getStaticType() { return EventType::APP_TICK; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "AppTick"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }

};

class AppUpdateEvent : public Event {
public:
    AppUpdateEvent() = default;

    [[nodiscard]] static EventType getStaticType() { return EventType::APP_UPDATE; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "AppUpdate"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }

};

class AppRenderEvent : public Event {
public:
    AppRenderEvent() = default;

    [[nodiscard]] static EventType getStaticType() { return EventType::APP_RENDER; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "AppRender"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_APPLICATION; }

};
}
