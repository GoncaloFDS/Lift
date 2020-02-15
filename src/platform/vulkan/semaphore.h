#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class Semaphore final {
public:
    Semaphore(const Semaphore &) = delete;
    Semaphore &operator=(const Semaphore &) = delete;
    Semaphore &operator=(Semaphore &&) = delete;

    explicit Semaphore(const Device &device);
    Semaphore(Semaphore &&other) noexcept;
    ~Semaphore();

    [[nodiscard]] VkSemaphore handle() const { return semaphore_; }
    [[nodiscard]] const Device &device() const { return device_; }

private:
    VkSemaphore semaphore_ {};

    const class Device &device_;
};

}  // namespace vulkan
