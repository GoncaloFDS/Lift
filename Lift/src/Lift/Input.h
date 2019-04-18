#pragma once

#include  "Lift/Core.h"

namespace Lift {

	class LIFT_API Input {
	public:
		virtual ~Input() = default;
		inline static bool IsKeyPressed(int keyCode);
		inline static bool IsMouseButtonPressed(int button);
		inline static std::pair<float, float> GetMousePos();
		inline static float GetMouseX();
		inline static float GetMouseY();

	protected:
		virtual bool IsKeyPressedImpl(int keyCode) = 0;
		virtual bool IsMouseButtonPressedImpl(int button) = 0;
		virtual std::pair<float, float> GetMousePosImpl() = 0;
		virtual float GetMouseXImpl() = 0;
		virtual float GetMouseYImpl() = 0;

	private:
		static Input* s_instance;
	};

	inline bool Input::IsKeyPressed(const int keyCode) {
		return s_instance->IsKeyPressedImpl(keyCode);
	}

	inline bool Input::IsMouseButtonPressed(int button) {
		return s_instance->IsMouseButtonPressedImpl(button);
	}

	inline std::pair<float, float> Input::GetMousePos() {
		return s_instance->GetMousePosImpl();
	}

	inline float Input::GetMouseX() {
		return s_instance->GetMouseXImpl();
	}

	inline float Input::GetMouseY() {
		return s_instance->GetMouseYImpl();
	}

}
