#include "pch.h"
#include  "WindowsInput.h"

#include "Lift/Application.h"
#include  <GLFW/glfw3.h>

namespace Lift {

	Input* Input::s_instance = new WindowsInput();

	bool Lift::WindowsInput::IsKeyPressedImpl(const int keyCode) {
		const auto window =static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		const auto state = glfwGetKey(window, keyCode);
		return state == GLFW_PRESS ||state == GLFW_REPEAT;
	}

	bool WindowsInput::IsMouseButtonPressedImpl(int button) {
		const auto window =static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		const auto state = glfwGetMouseButton(window, button);
		return state == GLFW_PRESS;
	}

	std::pair<float, float> WindowsInput::GetMousePosImpl() {
		const auto window =static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		double xPos, yPos;
		glfwGetCursorPos(window, &xPos, &yPos);

		return  { static_cast<float>(xPos), static_cast<float>(yPos) };
	}

	float WindowsInput::GetMouseXImpl() {
		auto[x, y] = GetMousePosImpl();
		return x;
	}

	float WindowsInput::GetMouseYImpl() {
		auto[x, y] = GetMousePosImpl();
		return y;
	}

}

