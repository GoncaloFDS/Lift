#pragma once

#include "event.h"
#include <sstream>

class WindowResizeEvent : public Event {
public:
    WindowResizeEvent(unsigned int width, unsigned int height) : width_(width), height_(height) {}

    [[nodiscard]] auto width() const -> unsigned int { return width_; }
    [[nodiscard]] auto height() const -> unsigned int { return height_; }

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << width_ << " x " << height_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::WINDOW_RESIZE; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "WindowResize"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }

private:
    unsigned int width_, height_;
};

class WindowCloseEvent : public Event {
public:
    WindowCloseEvent() = default;

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::WINDOW_CLOSE; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "WindowClose"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }
};

class WindowMinimizeEvent : public Event {
public:
    explicit WindowMinimizeEvent(bool is_minimized) : is_minimized_(is_minimized) {}

    [[nodiscard]] auto getIsMinimized() const -> bool { return is_minimized_; }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::WINDOW_MINIMIZE; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "WindowMinimize"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }

private:
    bool is_minimized_;
};

class AppTickEvent : public Event {
public:
    AppTickEvent() = default;

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::APP_TICK; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "AppTick"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }
};

class AppUpdateEvent : public Event {
public:
    AppUpdateEvent() = default;

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::APP_UPDATE; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "AppUpdate"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }
};

class AppRenderEvent : public Event {
public:
    AppRenderEvent() = default;

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::APP_RENDER; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "AppRender"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_APPLICATION; }
};
