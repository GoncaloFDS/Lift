#include "swap_chain.h"
#include "core.h"
#include "device.h"
#include "enumerate.h"
#include "image_view.h"
#include "instance.h"
#include "surface.h"
#include "window.h"

namespace vulkan {

SwapChain::SwapChain(const class Device& device, const bool vsync) :
    physical_device_(device.physicalDevice()), device_(device) {
    const auto details = querySwapChainSupport(device.physicalDevice(), device.surface().handle());
    LF_ASSERT(!details.formats.empty(), "[SwapChain] Image Format can't be empty");
    LF_ASSERT(!details.presentModes.empty(), "[SwapChain] Present Modes can't be empty");

    const auto& surface = device.surface();
    const auto& window = surface.instance().window();

    const auto surface_format = chooseSwapSurfaceFormat(details.formats);
    const auto present_mode = chooseSwapPresentMode(details.presentModes, vsync);
    const auto extent = chooseSwapExtent(window, details.capabilities);
    const auto image_count = chooseImageCount(details.capabilities);

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface.handle();
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.preTransform = details.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = nullptr;

    if (device.graphicsFamilyIndex() != device.presentFamilyIndex()) {
        uint32_t queue_family_indices[] = {device.graphicsFamilyIndex(), device.presentFamilyIndex()};

        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;      // Optional
        create_info.pQueueFamilyIndices = nullptr;  // Optional
    }

    vulkanCheck(vkCreateSwapchainKHR(device.handle(), &create_info, nullptr, &swap_chain_), "create swap chain!");

    min_image_count_ = details.capabilities.minImageCount;
    format_ = surface_format.format;
    extent_ = extent;
    images_ = getEnumerateVector(device_.handle(), swap_chain_, vkGetSwapchainImagesKHR);
    image_views_.reserve(images_.size());

    for (const auto image : images_) {
        image_views_.push_back(std::make_unique<ImageView>(device, image, format_, VK_IMAGE_ASPECT_COLOR_BIT));
    }
}

SwapChain::~SwapChain() {
    image_views_.clear();

    if (swap_chain_ != nullptr) {
        vkDestroySwapchainKHR(device_.handle(), swap_chain_, nullptr);
        swap_chain_ = nullptr;
    }
}

SwapChain::SupportDetails SwapChain::querySwapChainSupport(VkPhysicalDevice physical_device,
                                                           const VkSurfaceKHR surface) {
    SupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.capabilities);
    details.formats = getEnumerateVector(physical_device, surface, vkGetPhysicalDeviceSurfaceFormatsKHR);
    details.presentModes = getEnumerateVector(physical_device, surface, vkGetPhysicalDeviceSurfacePresentModesKHR);

    return details;
}

VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
        return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    LF_ASSERT(false, "[SwapChain] Found no suitable surface format");
    return {};
}

VkPresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& present_modes,
                                                  const bool vsync) {
    // VK_PRESENT_MODE_IMMEDIATE_KHR:
    //   Images submitted by your application are transferred to the screen right away, which may result in tearing.
    // VK_PRESENT_MODE_FIFO_KHR:
    //   The swap chain is a queue where the display takes an image from the front of the queue when the display is
    //   refreshed and the program inserts rendered images at the back of the queue. If the queue is full then the
    //   program has to wait. This is most similar to vertical sync as found in modern games. The moment that the
    //   display is refreshed is known as "vertical blank".
    // VK_PRESENT_MODE_FIFO_RELAXED_KHR:
    //   This mode only differs from the previous one if the application is late and the queue was empty at the last
    //   vertical blank. Instead of waiting for the next vertical blank, the image is transferred right away when it
    //   finally arrives. This may result in visible tearing.
    // VK_PRESENT_MODE_MAILBOX_KHR:
    //   This is another variation of the second mode. Instead of blocking the application when the queue is full, the
    //   images that are already queued are simply replaced with the newer ones.This mode can be used to implement
    //   triple buffering, which allows you to avoid tearing with significantly less latency issues than standard
    //   vertical sync that uses double buffering.

    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    if (std::find(present_modes.begin(), present_modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != present_modes.end()) {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }

    if (std::find(present_modes.begin(), present_modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != present_modes.end()) {
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    if (std::find(present_modes.begin(), present_modes.end(), VK_PRESENT_MODE_FIFO_RELAXED_KHR)
        != present_modes.end()) {
        return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChain::chooseSwapExtent(const Window& window, const VkSurfaceCapabilitiesKHR& capabilities) {
    // vulkan tells us to match the resolution of the window by setting the width and height in the currentExtent
    // member. However, some window managers do allow us to differ here and this is indicated by setting the width and
    // height in currentExtent to a special value: the maximum value of uint32_t. In that case we'll pick the resolution
    // that best matches the window within the minImageExtent and maxImageExtent bounds.
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    auto actual_extent = window.framebufferSize();

    actual_extent.width =
        std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
    actual_extent.height = std::max(capabilities.minImageExtent.height,
                                    std::min(capabilities.maxImageExtent.height, actual_extent.height));

    return actual_extent;
}

uint32_t SwapChain::chooseImageCount(const VkSurfaceCapabilitiesKHR& capabilities) {
    // The implementation specifies the minimum amount of images to function properly
    // and we'll try to have one more than that to properly implement triple buffering.
    // (tanguyf: or not, we can just rely on VK_PRESENT_MODE_MAILBOX_KHR with two buffers)
    uint32_t image_count = capabilities.minImageCount;  // +1;

    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    return image_count;
}

}  // namespace vulkan
