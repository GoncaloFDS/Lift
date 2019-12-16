#include "surface.h"
#include "instance.h"
#include "window.h"

namespace vulkan {

Surface::Surface(const class Instance& instance) :
    instance_(instance) {
    vulkanCheck(glfwCreateWindowSurface(instance.Handle(), instance.window().handle(), nullptr, &surface_),
                "create window surface");
}

Surface::~Surface() {
    if (surface_ != nullptr) {
        vkDestroySurfaceKHR(instance_.Handle(), surface_, nullptr);
        surface_ = nullptr;
    }
}

}
