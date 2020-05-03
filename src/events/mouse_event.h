#pragma once

#include "event.h"

class MouseMovedEvent : public Event {
public:
    MouseMovedEvent(const float x, const float y) : position_x_(x), position_y_(y) {}

    [[nodiscard]] auto x() const -> float { return position_x_; }
    [[nodiscard]] auto y() const -> float { return position_y_; }

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "MouseMovedEvent: " << position_x_ << ", " << position_y_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::MOUSE_MOVE; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "MouseMoved"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }

private:
    float position_x_, position_y_;
};

class MouseScrolledEvent : public Event {
public:
    MouseScrolledEvent(const float x_offset, const float y_offset) : x_offset_(x_offset), y_offset_(y_offset) {}

    [[nodiscard]] auto x() const -> float { return x_offset_; }
    [[nodiscard]] auto y() const -> float { return y_offset_; };

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "MouseScrolledEvent: " << x() << ", " << y();
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::MOUSE_SCROLLED; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "MouseScrolled"; }
    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }

private:
    float x_offset_, y_offset_;
};

class MouseButtonEvent : public Event {
public:
    [[nodiscard]] auto getMouseButton() const -> int { return button_; }

    [[nodiscard]] auto getCategoryFlags() const -> int override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }

protected:
    explicit MouseButtonEvent(const int button) : button_(button) {}

    int button_;
};

class MouseButtonPressedEvent : public MouseButtonEvent {
public:
    explicit MouseButtonPressedEvent(const int button) : MouseButtonEvent(button) {}

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "MouseButtonPressedEvent: " << button_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::MOUSE_BUTTON_PRESSED; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "MouseButtonPressed"; }
};

class MouseButtonReleasedEvent : public MouseButtonEvent {
public:
    explicit MouseButtonReleasedEvent(const int button) : MouseButtonEvent(button) {}

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "MouseButtonReleasedEvent: " << button_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::MOUSE_BUTTON_RELEASED; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "MouseButtonReleased"; }
};
