#include "pch.h"
#include  "windows_input.h"

#include "application.h"
#include  <GLFW/glfw3.h>

std::unique_ptr<lift::Input> lift::Input::k_Instance = std::make_unique<WindowsInput>();

bool lift::WindowsInput::isKeyPressedImpl(const int key_code) {
    const auto window = static_cast<GLFWwindow *>(Application::get().getWindow().getNativeWindow());
    const auto state = glfwGetKey(window, key_code);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool lift::WindowsInput::isMouseButtonPressedImpl(const int button) {
    const auto window = static_cast<GLFWwindow *>(Application::get().getWindow().getNativeWindow());
    const auto state = glfwGetMouseButton(window, button);
    return state == GLFW_PRESS;
}

std::pair<float, float> lift::WindowsInput::getMousePosImpl() {
    const auto window = static_cast<GLFWwindow *>(Application::get().getWindow().getNativeWindow());
    double x_pos, y_pos;
    glfwGetCursorPos(window, &x_pos, &y_pos);

    return {static_cast<float>(x_pos), static_cast<float>(y_pos)};
}

float lift::WindowsInput::getMouseXImpl() {
    auto[x, y] = getMousePosImpl();
    return x;
}

float lift::WindowsInput::getMouseYImpl() {
    auto[x, y] = getMousePosImpl();
    return y;
}
