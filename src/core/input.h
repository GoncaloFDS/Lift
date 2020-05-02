#pragma once
#include "core/glm.h"
#include <memory>
#include <unordered_map>
#include <utility>

class Input {
public:
    static bool isKeyPressed(int key_code);
    static bool isMouseButtonPressed(int key_code);

    static void registerKey(int key_code);
    static void unregisterKey(int key_code);

    static void moveMouse(float x, float y);

//    static const glm::vec2& mousePos() {return };
    static const glm::vec2& mouseDelta() { return mouse_delta_; };

private:
    static std::unique_ptr<Input> s_input_;
    static std::unordered_map<int, bool> pressed_keys_;

    static glm::vec2 last_mouse_position_;
    static glm::vec2 mouse_delta_;
};
