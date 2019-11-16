#include <pch.h>
#include "input.h"

glm::vec2 lift::Input::last_mouse_position_ = {0.0f, 0.0f};
glm::vec2 lift::Input::mouse_delta_ = {0.0f, 0.0f};

void lift::Input::onUpdate() {
    const auto x = getMouseX();
    const auto y = getMouseY();
    mouse_delta_ = {x - last_mouse_position_.x, y - last_mouse_position_.y};
    last_mouse_position_ = {x, y};
}

auto lift::Input::isKeyPressed(const int key_code) -> bool {
    return k_Instance->isKeyPressedImpl(key_code);
}

auto lift::Input::isMouseButtonPressed(const int button) -> bool {
    return k_Instance->isMouseButtonPressedImpl(button);
}

inline auto lift::Input::getMousePos() -> std::pair<float, float> {
    return k_Instance->getMousePosImpl();
}

inline auto lift::Input::getMouseX() -> float {
    return k_Instance->getMouseXImpl();
}

inline auto lift::Input::getMouseY() -> float {
    return k_Instance->getMouseYImpl();
}
