#pragma once

#include "core/os/window.h"
#include "GLFW/glfw3.h"

namespace lift {

class WindowsWindow : public Window {
public:
    WindowsWindow(const WindowProperties& props);
    virtual ~WindowsWindow();

    void onUpdate() override;

    [[nodiscard]] inline unsigned int width() const override;
    [[nodiscard]] inline unsigned int height() const override;

    [[nodiscard]] std::pair<int, int> getPosition() const override;

    //Window attributes
    inline void setEventCallback(const EventCallbackFn& callback) override;
    void setVSync(bool enabled) override;
    bool isVSync() const override;

    [[nodiscard]] inline void* getNativeWindow() const override;
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
