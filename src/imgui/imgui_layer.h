#pragma once
#include "core/utilities.h"
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
  VkExtent2D framebufferSize;
  float frameRate;
  float rayRate;
  uint32_t totalSamples;
};

class ImGuiData;
class ImguiLayer {
  public:
  ImguiLayer(vulkan::CommandPool &command_pool, const vulkan::SwapChain &swap_chain,
             const vulkan::DepthBuffer &depth_buffer, UserSettings &user_settings);
  ~ImguiLayer();

  void render(VkCommandBuffer command_buffer, const vulkan::FrameBuffer &frame_buffer, const Statistics &statistics);

  static bool wantsToCaptureKeyboard();
  static bool wantsToCaptureMouse();

  UserSettings &settings() { return user_settings_; }

  private:
  void drawSettings();
  void drawOverlay(const Statistics &statistics);

  std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
  std::unique_ptr<vulkan::RenderPass> render_pass_;
  UserSettings &user_settings_;
};
