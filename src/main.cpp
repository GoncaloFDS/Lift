#include <platform/vulkan/enumerate.h>
#include "pch.h"
#include "properties.h"
#include "application.h"
#include "vulkan/window.h"
#include "user_settings.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/texture.h"
#include "assets/uniform_buffer.h"

UserSettings createUserSettings(const Options& options);
void setVulkanDevice(lift::Application& application);

int main(int argc, const char* argv[]) noexcept {
    lift::Log::init();
    LF_WARN("Initialized Log");
    const Options options(argc, argv);
    const UserSettings user_settings = createUserSettings(options);
    const vulkan::WindowData window_properties{"Lift",
                                               options.width,
                                               options.height,
                                               options.benchmark && options.fullscreen,
                                               options.fullscreen,
                                               !options.fullscreen};

    lift::Application application(user_settings, window_properties, options.vSync);

    setVulkanDevice(application);

    application.run();

    return EXIT_SUCCESS;
}

UserSettings createUserSettings(const Options& options) {
    UserSettings user_settings{};

    user_settings.benchmark = options.benchmark;
    user_settings.benchmarkNextScenes = options.benchmarkNextScenes;
    user_settings.benchmarkMaxTime = options.benchmarkMaxTime;

    user_settings.sceneIndex = options.sceneIndex;

    user_settings.isRayTraced = true;
    user_settings.accumulateRays = true;
    user_settings.numberOfSamples = options.samples;
    user_settings.numberOfBounces = options.bounces;
    user_settings.maxNumberOfSamples = options.maxSamples;

    user_settings.showSettings = !options.benchmark;
    user_settings.showOverlay = true;

    return user_settings;
}

void setVulkanDevice(lift::Application& application) {
    const auto& physical_devices = application.physicalDevices();
    const auto result = std::find_if(physical_devices.begin(), physical_devices.end(), [](const VkPhysicalDevice& device) {
        VkPhysicalDeviceFeatures device_features;
        vkGetPhysicalDeviceFeatures(device, &device_features);

        if (!device_features.geometryShader) {
            return false;
        }

        const auto queue_families = vulkan::getEnumerateVector(device, vkGetPhysicalDeviceQueueFamilyProperties);
        const auto has_graphics_queue =
            std::find_if(queue_families.begin(), queue_families.end(), [](const VkQueueFamilyProperties& queue_family) {
                return queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT;
            });

        return has_graphics_queue != queue_families.end();
    });

    LF_ASSERT(result != physical_devices.end(), "Cannot find a suitable device");
    application.setPhysicalDevice(*result);
}

