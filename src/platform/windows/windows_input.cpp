#include "pch.h"
#include  "windows_input.h"

#include "application_old.h"
#include  <GLFW/glfw3.h>

std::unique_ptr<lift::Input> lift::Input::k_Instance = std::make_unique<WindowsInput>();

auto lift::WindowsInput::isKeyPressedImpl(const int key_code) -> bool {
    const auto window = static_cast<GLFWwindow*>(Application::get().getWindow().getNativeWindow());
    const auto state = glfwGetKey(window, key_code);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

auto lift::WindowsInput::isMouseButtonPressedImpl(const int button) -> bool {
    const auto window = static_cast<GLFWwindow*>(Application::get().getWindow().getNativeWindow());
    const auto state = glfwGetMouseButton(window, button);
    return state == GLFW_PRESS;
}

auto lift::WindowsInput::getMousePosImpl() -> std::pair<float, float> {
    const auto window = static_cast<GLFWwindow*>(Application::get().getWindow().getNativeWindow());
    double x_pos, y_pos;
    glfwGetCursorPos(window, &x_pos, &y_pos);

    return {static_cast<float>(x_pos), static_cast<float>(y_pos)};
}

auto lift::WindowsInput::getMouseXImpl() -> float {
    return getMousePosImpl().first;
}

auto lift::WindowsInput::getMouseYImpl() -> float {
    return getMousePosImpl().second;
}
