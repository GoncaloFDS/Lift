#pragma once

#include "window_properties.h"
#include "core/utilities.h"
#include <functional>
#include <vector>

namespace vulkan {

class Window final {
public:
    explicit Window(const WindowProperties& config);
    ~Window();

    [[nodiscard]] const WindowProperties& config() const { return config_; }
    [[nodiscard]] GLFWwindow* handle() const { return window_; }

    static std::vector<const char*> getRequiredInstanceExtensions() ;
    [[nodiscard]] float contentScale() const;
    static double time() ;
    [[nodiscard]] VkExtent2D framebufferSize() const;
    [[nodiscard]] VkExtent2D windowSize() const;

    std::function<void()> drawFrame;

    std::function<void(int key, int scancode, int action, int mods)> onKey;
    std::function<void(double xpos, double ypos)> onCursorPosition;
    std::function<void(int button, int action, int mods)> onMouseButton;

    void close() const;
    [[nodiscard]] bool isMinimized() const;
    void run() const;
    static void waitForEvents() ;

private:

    const WindowProperties config_;
    GLFWwindow* window_{};
};

}
