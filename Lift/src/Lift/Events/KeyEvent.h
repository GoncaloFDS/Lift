#pragma once

#include "Event.h"

 namespace Lift {

 	class LIFT_API KeyEvent : public Event {
	public:
		inline int GetKeyCode() const { return m_keyCode; }

 		EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
	protected:
		KeyEvent(int keycode)
			: m_keyCode(keycode) {}

 		int m_keyCode;
	};

 	class LIFT_API KeyPressedEvent : public KeyEvent {
	public:
		KeyPressedEvent(int keycode, int repeatCount)
			: KeyEvent(keycode), _repeatCount(repeatCount) {}

 		inline int GetRepeatCount() const { return _repeatCount; }

 		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << m_keyCode << " (" << _repeatCount << " repeats)";
			return ss.str();
		}

 		EVENT_CLASS_TYPE(KeyPressed)
	private:
		int _repeatCount;
	};

 	class LIFT_API KeyReleasedEvent : public KeyEvent {
	public:
		KeyReleasedEvent(int keycode)
			: KeyEvent(keycode) {}

 		std::string ToString() const override {
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << m_keyCode;
			return ss.str();
		}

 		EVENT_CLASS_TYPE(KeyReleased)
	};

	class LIFT_API KeyTypedEvent : public KeyEvent {
	public:
		KeyTypedEvent(int keycode)
			: KeyEvent(keycode) {}

 		std::string ToString() const override {
			std::stringstream ss;
			ss << "KeyTypedEvent: " << m_keyCode;
			return ss.str();
		}

 		EVENT_CLASS_TYPE(KeyTyped)
	};

} 