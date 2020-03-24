#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class Fence final {
public:
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence& operator=(Fence&&) = delete;

    explicit Fence(const Device& device, bool signaled);
    Fence(Fence&& other) noexcept;
    ~Fence();

    [[nodiscard]] const class Device& device() const { return device_; }
    [[nodiscard]] const VkFence& handle() const { return fence_; }

    void reset();
    void wait(uint64_t timeout) const;

private:
    const class Device& device_;

    VkFence fence_ {};
};

}  // namespace vulkan
