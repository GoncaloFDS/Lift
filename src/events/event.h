#pragma once

#include "pch.h"
#include "core.h"

namespace lift {
// Events are immediately dispatched
// TODO buffer events in an event bus and process them
// during the "event" part of the update stage

enum class EventType {
    NONE = 0,
    WINDOW_CLOSE,
    WINDOW_RESIZE,
    WINDOW_MINIMIZE,
    WINDOW_FOCUS,
    WINDOW_LOST_FOCUS,
    WINDOW_MOVED,
    APP_TICK,
    APP_UPDATE,
    APP_RENDER,
    KEY_PRESSED,
    KEY_RELEASED,
    KEY_TYPED,
    MOUSE_BUTTON_PRESSED,
    MOUSE_BUTTON_RELEASED,
    MOUSE_MOVE,
    MOUSE_SCROLLED
};

// Used to filter Events
enum EventCategory {
    NONE = 0,
    EVENT_CATEGORY_APPLICATION = bit(0),
    EVENT_CATEGORY_INPUT = bit(1),
    EVENT_CATEGORY_KEYBOARD = bit(2),
    EVENT_CATEGORY_MOUSE = bit(3),
    EVENT_CATEGORY_MOUSE_BUTTON = bit(4),
};

class Event {
public:
    bool handled = false;
public:
    virtual ~Event() = default;

    [[nodiscard]] virtual EventType getEventType() const = 0;
    [[nodiscard]] virtual const char* getName() const = 0;
    [[nodiscard]] virtual int getCategoryFlags() const = 0;
    [[nodiscard]] virtual std::string toString() const { return getName(); }

    [[nodiscard]] bool isInCategory(const EventCategory category) const {
        return getCategoryFlags() & category;
    }
};

class EventDispatcher {
    template<typename T>
    using EventFn = std::function<bool(T&)>;
public:
    EventDispatcher(Event& event)
        : event_(event) {
    }

    template<typename T>
    bool dispatch(EventFn<T> func) {
        if (event_.getEventType() == T::getStaticType()) {
            event_.handled = func(*static_cast<T*>(&event_));
            return true;
        }
        return false;
    }

private:
    Event& event_;
};

inline std::ostream& operator<<(std::ostream& os, const Event& e) {
    return os << e.toString();
}

}
