#pragma once

#include "event.h"

namespace lift {

class MouseMovedEvent : public Event {
public:
    MouseMovedEvent(const float x, const float y)
        : position_x_(x), position_y_(y) {
    }

    [[nodiscard]] float getX() const { return position_x_; }
    [[nodiscard]] float getY() const { return position_y_; }

    [[nodiscard]]  std::string toString() const override {
        std::stringstream ss;
        ss << "MouseMovedEvent: " << position_x_ << ", " << position_y_;
        return ss.str();
    }

    [[nodiscard]] static EventType getStaticType() { return EventType::MOUSE_MOVE; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "MouseMoved"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }
private:
    float position_x_, position_y_;
};

class MouseScrolledEvent : public Event {
public:
    MouseScrolledEvent(const float x_offset, const float y_offset)
        : x_offset_(x_offset), y_offset_(y_offset) {
    }

    [[nodiscard]]  float getXOffset() const { return x_offset_; }
    [[nodiscard]]  float getYOffset() const { return y_offset_; }

    [[nodiscard]]  std::string toString() const override {
        std::stringstream ss;
        ss << "MouseScrolledEvent: " << getXOffset() << ", " << getYOffset();
        return ss.str();
    }

    [[nodiscard]] static EventType getStaticType() { return EventType::MOUSE_SCROLLED; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "MouseScrolled"; }
    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }
private:
    float x_offset_, y_offset_;
};

class MouseButtonEvent : public Event {
public:
    [[nodiscard]]  int getMouseButton() const { return button_; }

    [[nodiscard]] int getCategoryFlags() const override { return EVENT_CATEGORY_MOUSE | EVENT_CATEGORY_INPUT; }
protected:
    MouseButtonEvent(const int button)
        : button_(button) {
    }

    int button_;
};

class MouseButtonPressedEvent : public MouseButtonEvent {
public:
    MouseButtonPressedEvent(const int button)
        : MouseButtonEvent(button) {
    }

    [[nodiscard]]  std::string toString() const override {
        std::stringstream ss;
        ss << "MouseButtonPressedEvent: " << button_;
        return ss.str();
    }

    [[nodiscard]] static EventType getStaticType() { return EventType::MOUSE_BUTTON_PRESSED; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "MouseButtonPressed"; }
};

class MouseButtonReleasedEvent : public MouseButtonEvent {
public:
    MouseButtonReleasedEvent(const int button)
        : MouseButtonEvent(button) {
    }

    [[nodiscard]]  std::string toString() const override {
        std::stringstream ss;
        ss << "MouseButtonReleasedEvent: " << button_;
        return ss.str();
    }

    [[nodiscard]] static EventType getStaticType() { return EventType::MOUSE_BUTTON_RELEASED; }
    [[nodiscard]] EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]] const char *getName() const override { return "MouseButtonReleased"; }
};

}
