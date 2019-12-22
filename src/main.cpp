
#include "platform/vulkan/enumerate.h"
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
    application.setPhysicalDevice(physical_devices[0]);
}

}
