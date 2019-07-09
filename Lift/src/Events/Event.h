#pragma once

#include "pch.h"
#include  "Core.h"

namespace lift {
	// Events are immediately dispatched
	// TODO buffer events in an event bus and process them 
	// during the "event" part of the update stage 

	enum class EventType {
		kNone = 0,
		kWindowClose,
		kWindowResize,
		kWindowMinimize,
		kWindowFocus,
		kWindowLostFocus,
		kWindowMoved,
		// 
		kAppTick,
		kAppUpdate,
		kAppRender,
		// Might not used this
		kKeyPressed,
		kKeyReleased,
		kKeyTyped,
		kMouseButtonPressed,
		kMouseButtonReleased,
		kMouseMoved,
		kMouseScrolled
	};

	// Used to filter Events
	enum EventCategory {
		kNone = 0,
		kEventCategoryApplication = Bit(0),
		kEventCategoryInput = Bit(1),
		kEventCategoryKeyboard = Bit(2),
		kEventCategoryMouse = Bit(3),
		kEventCategoryMouseButton = Bit(4),
	};

#define EVENT_CLASS_TYPE(type) static EventType GetStaticType() { return EventType::##type; }\
							virtual EventType GetEventType() const override { return GetStaticType(); }\
							virtual const char* GetName() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int GetCategoryFlags() const override { return category; }

	class Event {
	public:
		bool handled_ = false;
	public:
		virtual ~Event() = default;

		[[nodiscard]] virtual EventType GetEventType() const = 0;
		[[nodiscard]] virtual const char* GetName() const = 0;
		[[nodiscard]] virtual int GetCategoryFlags() const = 0;
		[[nodiscard]] virtual std::string ToString() const { return GetName(); }

		bool IsInCategory(const EventCategory category) const {
			return GetCategoryFlags() & category;

		}
	};

	class EventDispatcher {
		template <typename T>
		using EventFn = std::function<bool(T&)>;
	public:
		EventDispatcher(Event& event)
			: event_(event) {
		}

		template <typename T>
		bool Dispatch(EventFn<T> func) {
			if (event_.GetEventType() == T::GetStaticType()) {
				event_.handled_ = func(*static_cast<T*>(&event_));
				return true;
			}
			return false;
		}

	private:
		Event& event_;
	};

	inline std::ostream& operator<<(std::ostream& os, const Event& e) {
		return os << e.ToString();
	}

}
