#pragma once

#include "Event.h"

namespace lift {
	class WindowResizeEvent : public Event {
	public:
		WindowResizeEvent(unsigned int width, unsigned int height)
			: width_(width), height_(height) {
		}

		unsigned int GetWidth() const { return width_; }
		unsigned int GetHeight() const { return height_; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "WindowResizeEvent: " << width_ << " x " << height_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kWindowResize)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)

	private:
		unsigned int width_, height_;
	};

	class WindowCloseEvent : public Event {
	public:
		WindowCloseEvent() = default;

		EVENT_CLASS_TYPE(kWindowClose)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)
	};

	class WindowMinimizeEvent : public Event {
	public:
		WindowMinimizeEvent(bool is_minimized)
			: is_minimized_(is_minimized) {
		}

		bool IsMinimized() const { return is_minimized_; }

		EVENT_CLASS_TYPE(kWindowMinimize)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)
	private:
		bool is_minimized_;
	};

	class AppTickEvent : public Event {
	public:
		AppTickEvent() = default;

		EVENT_CLASS_TYPE(kAppTick)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)

	};

	class AppUpdateEvent : public Event {
	public:
		AppUpdateEvent() = default;

		EVENT_CLASS_TYPE(kAppUpdate)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)

	};

	class AppRenderEvent : public Event {
	public:
		AppRenderEvent() = default;

		EVENT_CLASS_TYPE(kAppRender)
		EVENT_CLASS_CATEGORY(kEventCategoryApplication)

	};
}
