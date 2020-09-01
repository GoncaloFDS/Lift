#pragma once
#include "core/utilities.h"
#include <assets/camera.h>
#include <memory>

namespace vulkan {
class CommandPool;
class DepthBuffer;
class DescriptorPool;
class FrameBuffer;
class RenderPass;
class SwapChain;
}  // namespace vulkan

struct UserSettings;

struct Statistics final {
    VkExtent2D framebuffer_size;
    float frame_rate;
    float frame_time;
    float denoiser_time;
    float total_time;
    uint32_t total_samples;
};

class ImguiLayer {
public:
    ImguiLayer(vulkan::CommandPool& command_pool,
               const vulkan::SwapChain& swap_chain,
               const vulkan::DepthBuffer& depth_buffer,
               UserSettings& user_settings);
    ~ImguiLayer();

    void updateInfo(const Statistics& statistics, const CameraState& camera_state);
    void render(VkCommandBuffer command_buffer, const vulkan::FrameBuffer& frame_buffer);

    static bool wantsToCaptureKeyboard();
    static bool wantsToCaptureMouse();

    UserSettings& settings() { return user_settings_; }

private:
    void drawSettings(const CameraState& camera_state);
    void drawOverlay(const Statistics& statistics);

    std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
    std::unique_ptr<vulkan::RenderPass> render_pass_;
    UserSettings& user_settings_;
    Statistics statistics_;
    CameraState camera_state_;

};
