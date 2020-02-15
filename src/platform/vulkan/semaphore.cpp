#include "semaphore.h"
#include "device.h"

namespace vulkan {

Semaphore::Semaphore(const class Device &device) : device_(device) {
    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    vulkanCheck(vkCreateSemaphore(device.handle(), &semaphore_info, nullptr, &semaphore_), "create semaphores");
}

Semaphore::Semaphore(Semaphore &&other) noexcept : device_(other.device_), semaphore_(other.semaphore_) {
    other.semaphore_ = nullptr;
}

Semaphore::~Semaphore() {
    if (semaphore_ != nullptr) {
        vkDestroySemaphore(device_.handle(), semaphore_, nullptr);
        semaphore_ = nullptr;
    }
}

}  // namespace vulkan
