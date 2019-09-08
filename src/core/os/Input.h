#pragma once

#include  "Core.h"

namespace lift {

class Input {
public:
    virtual ~Input() = default;
    inline static void onUpdate();
    inline static bool isKeyPressed(int key_code);
    inline static bool isMouseButtonPressed(int button);
    inline static std::pair<float, float> getMousePos();
    inline static float getMouseX();
    inline static float getMouseY();
    inline static const vec2 &getMouseDelta() { return mouse_delta_; };

protected:
    virtual bool isKeyPressedImpl(int key_code) = 0;
    virtual bool isMouseButtonPressedImpl(int button) = 0;
    virtual std::pair<float, float> getMousePosImpl() = 0;
    virtual float getMouseXImpl() = 0;
    virtual float getMouseYImpl() = 0;

    inline static vec2 last_mouse_position_ = {0, 0};
    inline static vec2 mouse_delta_ = {0, 0};

private:
    static std::unique_ptr<Input> k_Instance;
};

inline void Input::onUpdate() {
    const auto x = getMouseX();
    const auto y = getMouseY();
    mouse_delta_ = {x - last_mouse_position_.x, y - last_mouse_position_.y};
    last_mouse_position_ = {x, y};
}

inline bool Input::isKeyPressed(const int key_code) {
    return k_Instance->isKeyPressedImpl(key_code);
}

inline bool Input::isMouseButtonPressed(const int button) {
    return k_Instance->isMouseButtonPressedImpl(button);
}

inline std::pair<float, float> Input::getMousePos() {
    return k_Instance->getMousePosImpl();
}

inline float Input::getMouseX() {
    return k_Instance->getMouseXImpl();
}

inline float Input::getMouseY() {
    return k_Instance->getMouseYImpl();
}

}
