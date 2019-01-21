#pragma once

#include "Event.h"

 namespace Lift {

 	class LIFT_API KeyEvent : public Event {
	public:
		inline int GetKeyCode() const { return _keyCode; }

 		EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)
	protected:
		KeyEvent(int keycode)
			: _keyCode(keycode) {}

 		int _keyCode;
	};

 	class LIFT_API KeyPressedEvent : public KeyEvent {
	public:
		KeyPressedEvent(int keycode, int repeatCount)
			: KeyEvent(keycode), _repeatCount(repeatCount) {}

 		inline int GetRepeatCount() const { return _repeatCount; }

 		std::string ToString() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << _keyCode << " (" << _repeatCount << " repeats)";
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
			ss << "KeyReleasedEvent: " << _keyCode;
			return ss.str();
		}

 		EVENT_CLASS_TYPE(KeyReleased)
	};
} 