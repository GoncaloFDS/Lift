#include "pch.h"
#include  "WindowsInput.h"

#include "Application.h"
#include  <GLFW/glfw3.h>

std::unique_ptr<lift::Input> lift::Input::instance_ = std::make_unique<WindowsInput>();

bool lift::WindowsInput::IsKeyPressedImpl(const int key_code) {
	const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
	const auto state = glfwGetKey(window, key_code);
	return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool lift::WindowsInput::IsMouseButtonPressedImpl(const int button) {
	const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
	const auto state = glfwGetMouseButton(window, button);
	return state == GLFW_PRESS;
}

std::pair<float, float> lift::WindowsInput::GetMousePosImpl() {
	const auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
	double x_pos, y_pos;
	glfwGetCursorPos(window, &x_pos, &y_pos);

	return {static_cast<float>(x_pos), static_cast<float>(y_pos)};
}

float lift::WindowsInput::GetMouseXImpl() {
	auto [x, y] = GetMousePosImpl();
	return x;
}

float lift::WindowsInput::GetMouseYImpl() {
	auto [x, y] = GetMousePosImpl();
	return y;
}
