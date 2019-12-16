#pragma once

#include "VulkanError.h"

namespace vulkan {
class Device;

class Semaphore final {
public:

    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore& operator=(Semaphore&&) = delete;

    explicit Semaphore(const Device& device);
    Semaphore(Semaphore&& other) noexcept;
    ~Semaphore();

    [[nodiscard]] const Device& device() const { return device_; }

private:

    const class Device& device_;

VULKAN_HANDLE(VkSemaphore, semaphore_)
};

}
