#pragma once

#include "core/os/input.h"

namespace lift {
class WindowsInput : public Input {
protected:
    bool isKeyPressedImpl(int key_code) override;
    bool isMouseButtonPressedImpl(int button) override;
    std::pair<float, float> getMousePosImpl() override;
    float getMouseXImpl() override;
    float getMouseYImpl() override;
};

}
