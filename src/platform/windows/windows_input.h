#pragma once

#include "core/input.h"

namespace lift {
class WindowsInput : public Input {
protected:
    auto isKeyPressedImpl(int key_code) -> bool override;
    auto isMouseButtonPressedImpl(int button) -> bool override;
    auto getMousePosImpl() -> std::pair<float, float> override;
    auto getMouseXImpl() -> float override;
    auto getMouseYImpl() -> float override;
};

}
