#include "pch.h"
#include  "input.h"

std::unordered_map<int, bool> lift::Input::pressed_keys_{};

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
