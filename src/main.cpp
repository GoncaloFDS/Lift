

#include "application.h"
#include "assets/scene.h"
#include "assets/texture.h"
#include "assets/uniform_buffer.h"
#include "options.h"
#include "vulkan/enumerate.h"
#include "vulkan/window.h"

UserSettings createUserSettings(const Options& options);
void setVulkanDevice(Application& application);

int main(int argc, const char* argv[]) noexcept {
    Log::init();
    const Options options(argc, argv);
    const UserSettings user_settings = createUserSettings(options);
    const vulkan::WindowData window_properties {"Lift",
                                                options.width,
                                                options.height,
                                                options.benchmark && options.fullscreen,
                                                options.fullscreen,
                                                !options.fullscreen};

    Application application(user_settings, window_properties);

    setVulkanDevice(application);

    application.run();

    return EXIT_SUCCESS;
}

UserSettings createUserSettings(const Options& options) {
    UserSettings user_settings {};

    user_settings.scene_index = options.scene_index;
    user_settings.algorithm_index = options.algorithm_index;

    user_settings.accumulate_rays = true;
    user_settings.is_denoised = false;
    user_settings.enable_mis = true;
    user_settings.number_of_samples = options.samples;
    user_settings.number_of_bounces = options.bounces;
    user_settings.max_number_of_samples = options.max_samples;

    user_settings.show_settings = !options.benchmark;
    user_settings.show_overlay = true;

    return user_settings;
}

void setVulkanDevice(Application& application) {
    const auto& physical_devices = application.physicalDevices();
    const auto result =
        std::find_if(physical_devices.begin(), physical_devices.end(), [](const VkPhysicalDevice& device) {
            VkPhysicalDeviceFeatures device_features;
            vkGetPhysicalDeviceFeatures(device, &device_features);

            if (!device_features.geometryShader) {
                return false;
            }

            const auto queue_families = vulkan::getEnumerateVector(device, vkGetPhysicalDeviceQueueFamilyProperties);
            const auto has_graphics_queue =
                std::find_if(queue_families.begin(),
                             queue_families.end(),
                             [](const VkQueueFamilyProperties& queue_family) {
                                 return queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT;
                             });

            return has_graphics_queue != queue_families.end();
        });

    LF_ASSERT(result != physical_devices.end(), "Cannot find a suitable device");
    application.setPhysicalDevice(*result);
}
