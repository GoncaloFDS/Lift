#pragma once

#include "core.h"

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
    NONE = 0u,
    EVENT_CATEGORY_APPLICATION = bit(0u),
    EVENT_CATEGORY_INPUT = bit(1u),
    EVENT_CATEGORY_KEYBOARD = bit(2u),
    EVENT_CATEGORY_MOUSE = bit(3u),
    EVENT_CATEGORY_MOUSE_BUTTON = bit(4u),
};

class Event {
public:
    bool handled = false;

public:
    virtual ~Event() = default;

    [[nodiscard]] virtual auto getEventType() const -> EventType = 0;
    [[nodiscard]] virtual auto getName() const -> const char * = 0;
    [[nodiscard]] virtual auto getCategoryFlags() const -> int = 0;
    [[nodiscard]] virtual auto toString() const -> std::string { return getName(); }

    [[nodiscard]] auto isInCategory(const EventCategory category) const -> bool {
        return getCategoryFlags() & category;
    }
};

class EventDispatcher {
    template<typename T>
    using EventFn = std::function<bool(T &)>;

public:
    explicit EventDispatcher(Event &event) : event_(event) {}

    template<typename T>
    auto dispatch(EventFn<T> func) -> bool {
        if (event_.getEventType() == T::getStaticType()) {
            event_.handled = func(*static_cast<T *>(&event_));
            return true;
        }
        return false;
    }

private:
    Event &event_;
};

inline auto operator<<(std::ostream &os, const Event &e) -> std::ostream & {
    return os << e.toString();
}
