#include <pch.h>
#include "window.h"
#include "core/stb_image_impl.h"
#include "event.h"
#include "key_event.h"
#include "mouse_event.h"
#include "application_event.h"
#include "core/input.h"

namespace vulkan {

void glfwErrorCallback(const int error, const char* const description) {
    LF_ERROR("GLFW: {0} -> code: '{1}'", description, error);
}

Window::Window(const WindowData& config) :
    config_(config) {
    glfwSetErrorCallback(glfwErrorCallback);

    LF_INFO("Creating window {0} -> {1}x{2}", config.title, config.width, config.height);
    if (!glfwInit()) {
        LF_ASSERT(false, "glfwInit failed");
        return;
    }

    if (!glfwVulkanSupported()) {
        LF_ASSERT(false, "Vulkan is not supported by glfw context")
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);

    const auto monitor = config.fullscreen ? glfwGetPrimaryMonitor() : nullptr;

    window_ = glfwCreateWindow(config.width, config.height, config.title.c_str(), monitor, nullptr);
    LF_ASSERT(window_, "Failed to create a Window");

    if (config.cursorDisabled) {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    glfwSetWindowUserPointer(window_, &config_);

    glfwSetWindowSizeCallback(window_, [](GLFWwindow* window, int width, int height) {
        WindowData& config = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
        config.width = width;
        config.height = height;

        WindowResizeEvent event(width, height);
        config.eventCallbackFn(event);
    });

    glfwSetWindowCloseCallback(window_, [](GLFWwindow* window) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
        WindowCloseEvent event;
        data.eventCallbackFn(event);
    });

    glfwSetWindowIconifyCallback(window_, [](GLFWwindow* window, int iconified) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        WindowMinimizeEvent event(iconified);
        data.eventCallbackFn(event);
    });

    glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int scan_code, const int action, int mods) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        switch (action) {
            case GLFW_PRESS: {
                KeyPressedEvent event(key, 0);
                data.eventCallbackFn(event);
                break;
            }
            case GLFW_RELEASE: {
                KeyReleasedEvent event(key);
                data.eventCallbackFn(event);
                break;
            }
            case GLFW_REPEAT: {
                KeyPressedEvent event(key, 1);
                data.eventCallbackFn(event);
                break;
            }
            default:;
        }
    });

    glfwSetCharCallback(window_, [](GLFWwindow* window, const uint32_t key_code) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        KeyTypedEvent event(key_code);
        data.eventCallbackFn(event);
    });

    glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button, int action, int mods) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        switch (action) {
            case GLFW_PRESS: {
                Input::registerKey(button);
                MouseButtonPressedEvent event(button);
                data.eventCallbackFn(event);
                break;
            }
            case GLFW_RELEASE: {
                Input::unregisterKey(button);
                MouseButtonReleasedEvent event(button);
                data.eventCallbackFn(event);
                break;
            }
        }
    });

    glfwSetScrollCallback(window_, [](GLFWwindow* window, const double x_offset, const double y_offset) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        MouseScrolledEvent event(static_cast<float>(x_offset), static_cast<float>(y_offset));
        data.eventCallbackFn(event);
    });

    glfwSetCursorPosCallback(window_, [](GLFWwindow* window, const double x_pos, const double y_pos) {
        WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

        MouseMovedEvent event(static_cast<float>(x_pos), static_cast<float>(y_pos));
        data.eventCallbackFn(event);
    });
}

Window::~Window() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
    glfwSetErrorCallback(nullptr);
}

std::vector<const char*> Window::getRequiredInstanceExtensions() {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    return std::vector<const char*>(glfw_extensions, glfw_extensions + glfw_extension_count);
}

float Window::contentScale() const {
    float xscale;
    float yscale;
    glfwGetWindowContentScale(window_, &xscale, &yscale);

    return xscale;
}

double Window::time() {
    return glfwGetTime();
}

VkExtent2D Window::framebufferSize() const {
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    return VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

VkExtent2D Window::windowSize() const {
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    return VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

void Window::close() const {
    glfwSetWindowShouldClose(window_, 1);
}

bool Window::isMinimized() const {
    const auto size = framebufferSize();
    return size.height == 0 && size.width == 0;
}

void Window::run() const {
    glfwSetTime(0.0);

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        if (drawFrame) {
            drawFrame();
        }
    }
}

void Window::waitForEvents() {
    glfwWaitEvents();
}

void Window::setEventCallbackFn(const EventCallbackFn& callback) {
    config_.eventCallbackFn = callback;
}

}
