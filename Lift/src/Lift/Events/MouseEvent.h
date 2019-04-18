#pragma once

#include "Event.h"

namespace Lift {

	class LIFT_API MouseMovedEvent : public Event {
	public:
		MouseMovedEvent(float x, float y)
			: _mouseX(x), _mouseY(y) {
		}

		inline float GetX() const { return _mouseX; }
		inline float GetY() const { return _mouseY; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseMovedEvent: " << _mouseX << ", " << _mouseY;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseMoved)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float _mouseX, _mouseY;
	};

	class LIFT_API MouseScrolledEvent : public Event {
	public:
		MouseScrolledEvent(float xOffset, float yOffset)
			: _xOffset(xOffset), _yOffset(yOffset) {
		}

		inline float GetXOffset() const { return _xOffset; }
		inline float GetYOffset() const { return _yOffset; }

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseScrolledEvent: " << GetXOffset() << ", " << GetYOffset();
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseScrolled)
		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	private:
		float _xOffset, _yOffset;
	};

	class LIFT_API MouseButtonEvent : public Event {
	public:
		inline int GetMouseButton() const { return m_button; }

		EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)
	protected:
		MouseButtonEvent(int button)
			: m_button(button) {
		}

		int m_button;
	};

	class LIFT_API MouseButtonPressedEvent : public MouseButtonEvent {
	public:
		MouseButtonPressedEvent(int button)
			: MouseButtonEvent(button) {
		}

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << m_button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonPressed)
	};

	class LIFT_API MouseButtonReleasedEvent : public MouseButtonEvent {
	public:
		MouseButtonReleasedEvent(int button)
			: MouseButtonEvent(button) {
		}

		std::string ToString() const override {
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << m_button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseButtonReleased)
	};

}
