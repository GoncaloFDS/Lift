#pragma once

#include "Event.h"

namespace lift {

	class MouseMovedEvent : public Event {
	public:
		MouseMovedEvent(float x, float y)
			: position_x_(x), position_y_(y) {
		}

		float GetX() const { return position_x_; }
		float GetY() const { return position_y_; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseMovedEvent: " << position_x_ << ", " << position_y_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kMouseMoved)
		EVENT_CLASS_CATEGORY(kEventCategoryMouse | kEventCategoryInput)
	private:
		float position_x_, position_y_;
	};

	class MouseScrolledEvent : public Event {
	public:
		MouseScrolledEvent(const float x_offset, const float y_offset)
			: x_offset_(x_offset), y_offset_(y_offset) {
		}

		float GetXOffset() const { return x_offset_; }
		float GetYOffset() const { return y_offset_; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseScrolledEvent: " << GetXOffset() << ", " << GetYOffset();
			return ss.str();
		}

		EVENT_CLASS_TYPE(kMouseScrolled)
		EVENT_CLASS_CATEGORY(kEventCategoryMouse | kEventCategoryInput)
	private:
		float x_offset_, y_offset_;
	};

	class MouseButtonEvent : public Event {
	public:
		int GetMouseButton() const { return button_; }

		EVENT_CLASS_CATEGORY(kEventCategoryMouse | kEventCategoryInput)
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

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << button_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kMouseButtonPressed)
	};

	class MouseButtonReleasedEvent : public MouseButtonEvent {
	public:
		MouseButtonReleasedEvent(const int button)
			: MouseButtonEvent(button) {
		}

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << button_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kMouseButtonReleased)
	};

}
