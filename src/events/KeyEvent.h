#pragma once

#include "Event.h"

namespace lift {

class KeyEvent : public Event {
public:
    [[nodiscard]] unsigned int getKeyCode() const { return key_code_; }

    [[nodiscard]]int getCategoryFlags() const override { return EVENT_CATEGORY_KEYBOARD | EVENT_CATEGORY_INPUT; }
protected:
    KeyEvent(const int keycode)
        : key_code_(keycode) {
    }

    unsigned int key_code_;
};

class KeyPressedEvent : public KeyEvent {
public:
    KeyPressedEvent(const int keycode, const int repeat_count)
        : KeyEvent(keycode), repeat_count_(repeat_count) {
    }

    int getRepeatCount() const { return repeat_count_; }

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "KeyPressedEvent: " << key_code_ << " (" << repeat_count_ << " repeats)";
        return ss.str();
    }

    [[nodiscard]]static EventType getStaticType() { return EventType::KEY_PRESSED; }
    [[nodiscard]]EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]]const char *getName() const override { return "KEY_PRESSED"; }
private:
    int repeat_count_;
};

class KeyReleasedEvent : public KeyEvent {
public:
    KeyReleasedEvent(const int keycode)
        : KeyEvent(keycode) {
    }

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "KeyReleasedEvent: " << key_code_;
        return ss.str();
    }

    [[nodiscard]]static EventType getStaticType() { return EventType::KEY_RELEASED; }
    [[nodiscard]]EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]]const char *getName() const override { return "KEY_RELEASED"; }
};

class KeyTypedEvent : public KeyEvent {
public:
    explicit KeyTypedEvent(const int keycode)
        : KeyEvent(keycode) {
    }

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "KeyTypedEvent: " << key_code_;
        return ss.str();
    }

    [[nodiscard]]static EventType getStaticType() { return EventType::KEY_TYPED; }
    [[nodiscard]]EventType getEventType() const override { return getStaticType(); }
    [[nodiscard]]const char *getName() const override { return "KEY_TYPED"; }
};

}
