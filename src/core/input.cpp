
#include "input.h"

glm::vec2 Input::last_mouse_position_ = {0.0f, 0.0f};
glm::vec2 Input::mouse_delta_ = {0.0f, 0.0f};
std::unordered_map<int, bool> Input::pressed_keys_ {};

bool Input::isKeyPressed(int key_code) {
    return pressed_keys_[key_code];
}

bool Input::isMouseButtonPressed(int key_code) {
    return pressed_keys_[key_code];
}

void Input::registerKey(int key_code) {
    pressed_keys_[key_code] = true;
}

void Input::unregisterKey(int key_code) {
    pressed_keys_[key_code] = false;
}
void Input::moveMouse(float x, float y) {
    mouse_delta_ = {last_mouse_position_.x - x, last_mouse_position_.y - y};
    last_mouse_position_ = {x, y};
}
