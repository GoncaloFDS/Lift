#pragma once

#include "core/utilities.h"
#include <vector>

namespace vulkan {
class Surface;

class Device final {
public:
    Device(VkPhysicalDevice physical_device, const Surface &surface);
    ~Device();

    [[nodiscard]] VkDevice handle() const { return device_; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
    [[nodiscard]] const class Surface &surface() const { return surface_; }

    [[nodiscard]] uint32_t graphicsFamilyIndex() const { return graphics_family_index_; }
    [[nodiscard]] uint32_t computeFamilyIndex() const { return compute_family_index_; }
    [[nodiscard]] uint32_t presentFamilyIndex() const { return present_family_index_; }
    [[nodiscard]] uint32_t transferFamilyIndex() const { return transfer_family_index_; }
    [[nodiscard]] VkQueue graphicsQueue() const { return graphics_queue_; }
    [[nodiscard]] VkQueue computeQueue() const { return compute_queue_; }
    [[nodiscard]] VkQueue presentQueue() const { return present_queue_; }
    [[nodiscard]] VkQueue transferQueue() const { return transfer_queue_; }

    void waitIdle() const;

private:
    static void checkRequiredExtensions(VkPhysicalDevice physical_device);

    static const std::vector<const char *> required_extensions_;

    const VkPhysicalDevice physical_device_;
    const class Surface &surface_;

private:
    VkDevice device_ {};

    uint32_t graphics_family_index_ {};
    uint32_t compute_family_index_ {};
    uint32_t present_family_index_ {};
    uint32_t transfer_family_index_ {};

    VkQueue graphics_queue_ {};
    VkQueue compute_queue_ {};
    VkQueue present_queue_ {};
    VkQueue transfer_queue_ {};
};

}  // namespace vulkan
