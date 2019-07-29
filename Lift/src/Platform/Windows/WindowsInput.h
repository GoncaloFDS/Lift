#pragma once

#include "Core/os/Input.h"

namespace lift {
	class WindowsInput : public Input {
	protected:
		bool IsKeyPressedImpl(int key_code) override;
		bool IsMouseButtonPressedImpl(int button) override;
		std::pair<float, float> GetMousePosImpl() override;
		float GetMouseXImpl() override;
		float GetMouseYImpl() override;
	};

}
