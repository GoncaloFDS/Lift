#pragma once

#include "Event.h"

namespace lift {

	class KeyEvent : public Event {
	public:
		unsigned int GetKeyCode() const { return key_code_; }

		EVENT_CLASS_CATEGORY(kEventCategoryKeyboard | kEventCategoryInput)
	protected:
		KeyEvent(const int keycode)
			: key_code_(keycode) {
		}

		unsigned int key_code_;
	};

	class KeyPressedEvent : public KeyEvent {
	public:
		KeyPressedEvent(const int keycode, const int repeatCount)
			: KeyEvent(keycode), repeat_count_(repeatCount) {
		}

		int GetRepeatCount() const { return repeat_count_; }

		[[nodiscard]] std::string ToString() const override {
			std::stringstream ss;
			ss << "KeyPressedEvent: " << key_code_ << " (" << repeat_count_ << " repeats)";
			return ss.str();
		}

		EVENT_CLASS_TYPE(kKeyPressed)
	private:
		int repeat_count_;
	};

	class KeyReleasedEvent : public KeyEvent {
	public:
		KeyReleasedEvent(const int keycode)
			: KeyEvent(keycode) {
		}

		std::string ToString() const override {
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << key_code_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kKeyReleased)
	};

	class KeyTypedEvent : public KeyEvent {
	public:
		explicit KeyTypedEvent(const int keycode)
			: KeyEvent(keycode) {
		}

		std::string ToString() const override {
			std::stringstream ss;
			ss << "KeyTypedEvent: " << key_code_;
			return ss.str();
		}

		EVENT_CLASS_TYPE(kKeyTyped)
	};

}
