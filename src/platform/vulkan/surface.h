#pragma once

#include "VulkanError.h"

namespace vulkan {
class Instance;
class Window;

class Surface final {
public:
    explicit Surface(const Instance& instance);
    ~Surface();

    [[nodiscard]] const Instance& instance() const { return instance_; }

private:

    const class Instance& instance_;

VULKAN_HANDLE(VkSurfaceKHR, surface_)
};

}
