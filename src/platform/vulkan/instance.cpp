#include "instance.h"
#include "enumerate.h"
#include "window.h"
#include <algorithm>

namespace vulkan {

Instance::Instance(const class Window& window, const std::vector<const char*>& validation_layers) :
    window_(window),
    validation_layers_(validation_layers) {

    auto extensions = window.getRequiredInstanceExtensions();

    checkVulkanValidationLayerSupport(validation_layers);

    if (!validation_layers.empty()) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Lift";
    app_info.pEngineName = "Lift Engine";

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();

    vulkanCheck(vkCreateInstance(&create_info, nullptr, &instance_), "create instance");

    getVulkanDevices();
    getVulkanExtensions();
}

Instance::~Instance() {
    if (instance_ != nullptr) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = nullptr;
    }
}

void Instance::getVulkanDevices() {
    getEnumerateVector(instance_, vkEnumeratePhysicalDevices, physical_devices_);

    if (physical_devices_.empty()) {
//		Throw(std::runtime_error("found no vulkan physical devices"));
    }
}

void Instance::getVulkanExtensions() {
    getEnumerateVector(static_cast<const char*>(nullptr), vkEnumerateInstanceExtensionProperties, extensions_);
}

void Instance::checkVulkanValidationLayerSupport(const std::vector<const char*>& validation_layers) {
    const auto available_layers = getEnumerateVector(vkEnumerateInstanceLayerProperties);

    for (const char* layer : validation_layers) {
        auto result = std::find_if(available_layers.begin(),
                                   available_layers.end(),
                                   [layer](const VkLayerProperties& layer_properties) {
                                       return strcmp(layer, layer_properties.layerName) == 0;
                                   });

        if (result == available_layers.end()) {
//			Throw(std::runtime_error("could not find the requested validation layer: '" + std::string(layer) + "'"));
        }
    }
}

}
