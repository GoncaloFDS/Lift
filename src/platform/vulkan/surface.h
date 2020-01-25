#pragma once

#include "core/utilities.h"

namespace vulkan {
class Instance;
class Window;

class Surface final {
public:
    explicit Surface(const Instance& instance);
    ~Surface();

    [[nodiscard]] VkSurfaceKHR handle() const { return surface_; }
    [[nodiscard]] const Instance& instance() const { return instance_; }

private:
    VkSurfaceKHR surface_{};
    const class Instance& instance_;

};

}
