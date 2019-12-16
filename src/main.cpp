
#include "platform/vulkan/enumerate.h"
#include "Utilities/Console.hpp"
#include "properties.h"
#include "RayTracer.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace {
UserSettings createUserSettings(const Options& options);
void setVulkanDevice(vulkan::Application& application);
}

int main(int argc, const char* argv[]) noexcept {
    try {
        const Options options(argc, argv);
        const UserSettings user_settings = createUserSettings(options);
        const vulkan::WindowProperties window_properties{
            "Lift",
            options.width,
            options.height,
            options.benchmark && options.fullscreen,
            options.fullscreen,
            !options.fullscreen
        };

        RayTracer application(user_settings, window_properties, options.vSync);

        setVulkanDevice(application);

        application.run();

        return EXIT_SUCCESS;
    }

    catch (const std::exception& exception) {
    }

    catch (...) {
        Utilities::Console::Write(Utilities::Severity::Fatal, []() {
            std::cerr << "FATAL: caught unhandled exception" << std::endl;
        });
    }

    return EXIT_FAILURE;
}

namespace {

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

void setVulkanDevice(vulkan::Application& application) {
    const auto& physical_devices = application.physicalDevices();
    const auto
        result = std::find_if(physical_devices.begin(), physical_devices.end(), [](const VkPhysicalDevice& device) {
        // We want a device with geometry shader support.
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        if (!deviceFeatures.geometryShader) {
            return false;
        }

        // We want a device with a graphics queue.
        const auto queue_families = vulkan::getEnumerateVector(device, vkGetPhysicalDeviceQueueFamilyProperties);
        const auto has_graphics_queue =
            std::find_if(queue_families.begin(), queue_families.end(), [](const VkQueueFamilyProperties& queueFamily) {
                return queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT;
            });

        return has_graphics_queue != queue_families.end();
    });

    if (result == physical_devices.end()) {
//        Throw(std::runtime_error("cannot find a suitable device"));
    }

    application.setPhysicalDevice(*result);
}

}
