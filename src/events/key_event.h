#pragma once

#include "event.h"

namespace lift {

class KeyEvent : public Event {
public:
    [[nodiscard]] auto keyCode() const -> unsigned int { return key_code_; }
    [[nodiscard]] auto getCategoryFlags() const -> int override {
        return EVENT_CATEGORY_KEYBOARD | EVENT_CATEGORY_INPUT;
    }
protected:
    explicit KeyEvent(const int key_code)
        : key_code_(key_code) {
    }

    unsigned int key_code_;
};

class KeyPressedEvent : public KeyEvent {
public:
    KeyPressedEvent(const int key_code, const int repeat_count)
        : KeyEvent(key_code), repeat_count_(repeat_count) {
    }

    [[nodiscard]] auto getRepeatCount() const -> int { return repeat_count_; }

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "KeyPressedEvent: " << key_code_ << " (" << repeat_count_ << " repeats)";
        return ss.str();
    }

    [[nodiscard]]static auto getStaticType() -> EventType { return EventType::KEY_PRESSED; }
    [[nodiscard]]auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]]auto getName() const -> const char* override { return "KEY_PRESSED"; }
private:
    int repeat_count_;
};

class KeyReleasedEvent : public KeyEvent {
public:
    explicit KeyReleasedEvent(const int key_code)
        : KeyEvent(key_code) {
    }

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "KeyReleasedEvent: " << key_code_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::KEY_RELEASED; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "KEY_RELEASED"; }
};

class KeyTypedEvent : public KeyEvent {
public:
    explicit KeyTypedEvent(const int key_code)
        : KeyEvent(key_code) {
    }

    [[nodiscard]] auto toString() const -> std::string override {
        std::stringstream ss;
        ss << "KeyTypedEvent: " << key_code_;
        return ss.str();
    }

    [[nodiscard]] static auto getStaticType() -> EventType { return EventType::KEY_TYPED; }
    [[nodiscard]] auto getEventType() const -> EventType override { return getStaticType(); }
    [[nodiscard]] auto getName() const -> const char* override { return "KEY_TYPED"; }
};

}
