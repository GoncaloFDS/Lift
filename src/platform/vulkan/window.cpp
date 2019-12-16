#include "window.h"
#include "Utilities/StbImage.hpp"
#include <iostream>

namespace vulkan {

namespace {
void glfwErrorCallback(const int error, const char* const description) {
    std::cerr << "ERROR: GLFW: " << description << " (code: " << error << ")" << std::endl;
}

void glfwKeyCallback(GLFWwindow* window, const int key, const int scancode, const int action, const int mods) {
    const auto this_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if (this_->onKey) {
        this_->onKey(key, scancode, action, mods);
    }
}

void glfwCursorPositionCallback(GLFWwindow* window, const double xpos, const double ypos) {
    const auto this_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if (this_->onCursorPosition) {
        this_->onCursorPosition(xpos, ypos);
    }
}

void glfwMouseButtonCallback(GLFWwindow* window, const int button, const int action, const int mods) {
    const auto this_ = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if (this_->onMouseButton) {
        this_->onMouseButton(button, action, mods);
    }
}
}

Window::Window(const WindowProperties& config) :
    config_(config) {
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
//		Throw(std::runtime_error("glfwInit() failed"));
    }

    if (!glfwVulkanSupported()) {
//		Throw(std::runtime_error("glfwVulkanSupported() failed"));
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);

    const auto monitor = config.fullscreen ? glfwGetPrimaryMonitor() : nullptr;

    window_ = glfwCreateWindow(config.width, config.height, config.title.c_str(), monitor, nullptr);
    if (window_ == nullptr) {
//		Throw(std::runtime_error("failed to create window"));
    }

    GLFWimage icon;
    icon.pixels = stbi_load("../resources/textures/vulkan.png", &icon.width, &icon.height, nullptr, 4);
    if (icon.pixels == nullptr) {
//		Throw(std::runtime_error("failed to load icon"));
    }

    glfwSetWindowIcon(window_, 1, &icon);
    stbi_image_free(icon.pixels);

    if (config.cursorDisabled) {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, glfwKeyCallback);
    glfwSetCursorPosCallback(window_, glfwCursorPositionCallback);
    glfwSetMouseButtonCallback(window_, glfwMouseButtonCallback);
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
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
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

}
