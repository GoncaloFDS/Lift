#pragma once

#include "core/utilities.h"
#include <memory>
#include <vector>

namespace vulkan {
class Device;
class ImageView;
class Window;

class SwapChain final {
public:
    SwapChain(const Device &device, bool vsync);
    ~SwapChain();

    [[nodiscard]] VkSwapchainKHR handle() const { return swap_chain_; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
    [[nodiscard]] const class Device &device() const { return device_; }
    [[nodiscard]] uint32_t minImageCount() const { return min_image_count_; }
    [[nodiscard]] const std::vector<VkImage> &images() const { return images_; }
    [[nodiscard]] const std::vector<std::unique_ptr<ImageView>> &imageViews() const { return image_views_; }
    [[nodiscard]] const VkExtent2D &extent() const { return extent_; }
    [[nodiscard]] VkFormat format() const { return format_; }

private:
    struct SupportDetails {
        VkSurfaceCapabilitiesKHR capabilities {};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    static SupportDetails querySwapChainSupport(VkPhysicalDevice physical_device, VkSurfaceKHR surface);
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);
    static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &present_modes, bool vsync);
    static VkExtent2D chooseSwapExtent(const Window &window, const VkSurfaceCapabilitiesKHR &capabilities);
    static uint32_t chooseImageCount(const VkSurfaceCapabilitiesKHR &capabilities);

    const VkPhysicalDevice physical_device_;
    const class Device &device_;

    VkSwapchainKHR swap_chain_ {};

    uint32_t min_image_count_;
    VkFormat format_;
    VkExtent2D extent_ {};
    std::vector<VkImage> images_;
    std::vector<std::unique_ptr<ImageView>> image_views_;
};

}  // namespace vulkan
