#pragma once

#include <utility>
#include <memory>
#include "core.h"

namespace lift {
class Input {
public:
    virtual ~Input() = default;

    static void onUpdate();
    static auto isKeyPressed(int key_code) -> bool;
    static auto isMouseButtonPressed(int button) -> bool;

    inline static auto getMousePos() -> std::pair<float, float>;
    inline static auto getMouseX() -> float;
    inline static auto getMouseY() -> float;
    inline static auto getMouseDelta() -> const vec2& { return mouse_delta_; };

protected:
    virtual auto isKeyPressedImpl(int key_code) -> bool = 0;
    virtual auto isMouseButtonPressedImpl(int button) -> bool = 0;
    virtual auto getMousePosImpl() -> std::pair<float, float> = 0;
    virtual auto getMouseXImpl() -> float = 0;
    virtual auto getMouseYImpl() -> float = 0;

    static vec2 last_mouse_position_;
    static vec2 mouse_delta_;

private:
    static std::unique_ptr<Input> k_Instance;
};


}
