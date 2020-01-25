#pragma once
#include <utility>
#include <memory>
#include <unordered_map>

namespace lift {

class Input {
public:
    static bool isKeyPressed(int key_code);
    static bool isMouseButtonPressed(int key_code);

    static void registerKey(int key_code);
    static void unregisterKey(int key_code);

private:
    static std::unique_ptr<lift::Input> s_input_;
    static std::unordered_map<int, bool> pressed_keys_;
};

}
