#include "fence.h"
#include "device.h"

namespace vulkan {

Fence::Fence(const class Device& device, const bool signaled) :
    device_(device) {
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;

    vulkanCheck(vkCreateFence(device.Handle(), &fence_info, nullptr, &fence_),
                "create fence");
}

Fence::Fence(Fence&& other) noexcept :
    device_(other.device_),
    fence_(other.fence_) {
    other.fence_ = nullptr;
}

Fence::~Fence() {
    if (fence_ != nullptr) {
        vkDestroyFence(device_.Handle(), fence_, nullptr);
        fence_ = nullptr;
    }
}

void Fence::reset() {
    vulkanCheck(vkResetFences(device_.Handle(), 1, &fence_), "reset fence");
}

void Fence::wait(uint64_t timeout) const {
    vulkanCheck(vkWaitForFences(device_.Handle(), 1, &fence_, VK_TRUE, timeout), "wait for fence");
}

}
