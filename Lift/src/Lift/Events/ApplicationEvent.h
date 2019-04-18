#pragma once

#include "Event.h"

namespace Lift {
	class LIFT_API WindowResizeEvent : public Event {
	public:
		WindowResizeEvent(unsigned int width, unsigned int height)
			: _width(width), _height(height) {
		}

		inline unsigned int GetWidth() const { return _width; }
		inline unsigned int GetHeight() const { return _height; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "WindowResizeEvent: " << _width << " x " << _height;
			return ss.str();
		}

		EVENT_CLASS_TYPE(WindowResize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)

	private:
		unsigned int _width, _height;
	};

	class LIFT_API WindowCloseEvent : public Event {
	public:
		WindowCloseEvent() = default;

		EVENT_CLASS_TYPE(WindowClose)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class LIFT_API AppTickEvent : public Event {
	public:
		AppTickEvent() = default;

		EVENT_CLASS_TYPE(AppTick)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)

	};

	class LIFT_API AppUpdateEvent : public Event {
	public:
		AppUpdateEvent() = default;

		EVENT_CLASS_TYPE(AppUpdate)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)

	};

	class LIFT_API AppRenderEvent : public Event {
	public:
		AppRenderEvent() = default;

		EVENT_CLASS_TYPE(AppRender)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)

	};
}
