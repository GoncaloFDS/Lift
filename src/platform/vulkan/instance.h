#pragma once

#include "core/utilities.h"
#include <vector>

namespace vulkan {
class Window;

class Instance final {
public:
    Instance(const Window& window, const std::vector<const char*>& validation_layers);
    ~Instance();

    [[nodiscard]] const class Window& window() const { return window_; }

    [[nodiscard]] const std::vector<VkExtensionProperties>& extensions() const { return extensions_; }
    [[nodiscard]] const std::vector<VkPhysicalDevice>& physicalDevices() const { return physical_devices_; }
    [[nodiscard]] const std::vector<const char*>& validationLayers() const { return validation_layers_; }

private:

    void getVulkanDevices();
    void getVulkanExtensions();

    static void checkVulkanValidationLayerSupport(const std::vector<const char*>& validation_layers);

    const class Window& window_;
    const std::vector<const char*> validation_layers_;

VULKAN_HANDLE(VkInstance, instance_)

    std::vector<VkPhysicalDevice> physical_devices_;
    std::vector<VkExtensionProperties> extensions_;
    void checkVulkanMinimumVersion(const uint32_t min_version);
};

}
