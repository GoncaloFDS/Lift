#pragma once

#include "core/os/window.h"
#include "GLFW/glfw3.h"

namespace lift {

class WindowsWindow : public Window {
public:
    explicit WindowsWindow(const WindowProperties& props);
    ~WindowsWindow() override;

    void onUpdate() override;

    [[nodiscard]] inline auto width() const -> unsigned int override;
    [[nodiscard]] inline auto height() const -> unsigned int override;
    [[nodiscard]] inline auto size() const -> ivec2 override;
    [[nodiscard]] inline auto aspectRatio() const -> float override;

    [[nodiscard]] auto getPosition() const -> std::pair<int, int> override;

    //Window attributes
    inline void setEventCallback(const EventCallbackFn& callback) override;
    void setVSync(bool enabled) override;
    [[nodiscard]] auto isVSync() const -> bool override;

    [[nodiscard]] inline auto getNativeWindow() const -> void* override;
private:
    virtual void init(const WindowProperties& props);
    virtual void shutdown();

private:
    GLFWwindow* window_handle_{};

    struct WindowData {
        std::string title;
        unsigned int width{}, height{};
        unsigned int x{}, y{};
        bool v_sync{};

        EventCallbackFn event_callback;

    };

    WindowData properties_;
};

}
