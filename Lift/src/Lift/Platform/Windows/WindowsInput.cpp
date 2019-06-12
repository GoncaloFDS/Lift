#include "pch.h"
#include  "WindowsInput.h"

#include "Lift/Application.h"
#include  <GLFW/glfw3.h>

namespace lift {

	std::unique_ptr<Input> Input::instance_ = std::make_unique<WindowsInput>();

	bool lift::WindowsInput::IsKeyPressedImpl(const int key_code) {
		const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		const auto state = glfwGetKey(window, key_code);
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool WindowsInput::IsMouseButtonPressedImpl(const int button) {
		const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		const auto state = glfwGetMouseButton(window, button);
		return state == GLFW_PRESS;
	}

	std::pair<float, float> WindowsInput::GetMousePosImpl() {
		const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		double x_pos, y_pos;
		glfwGetCursorPos(window, &x_pos, &y_pos);

		return {static_cast<float>(x_pos), static_cast<float>(y_pos)};
	}

	float WindowsInput::GetMouseXImpl() {
		auto [x, y] = GetMousePosImpl();
		return x;
	}

	float WindowsInput::GetMouseYImpl() {
		auto [x, y] = GetMousePosImpl();
		return y;
	}

}
