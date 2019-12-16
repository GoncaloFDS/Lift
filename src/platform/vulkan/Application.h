#pragma once

#include "frame_buffer.h"
#include "window_properties.h"
#include <vector>
#include <memory>

namespace assets {
class Scene;
class UniformBufferObject;
class UniformBuffer;
}

namespace vulkan {
class Application {
public:
    virtual ~Application();

    [[nodiscard]] const std::vector<VkExtensionProperties>& extensions() const;
    [[nodiscard]] const std::vector<VkPhysicalDevice>& physicalDevices() const;

    void setPhysicalDevice(VkPhysicalDevice physical_device);
    void run();

protected:

    Application(const WindowProperties& window_properties, bool vsync, bool enable_validation_layers);

    [[nodiscard]] const class Window& window() const { return *window_; }
    [[nodiscard]] const class Device& device() const { return *device_; }
    class CommandPool& commandPool() { return *command_pool_; }
    [[nodiscard]] const class SwapChain& swapChain() const { return *swap_chain_; }
    [[nodiscard]] const class DepthBuffer& depthBuffer() const { return *depth_buffer_; }
    [[nodiscard]] const std::vector<assets::UniformBuffer>& uniformBuffers() const { return uniform_buffers_; }
    [[nodiscard]] const class GraphicsPipeline& graphicsPipeline() const { return *graphics_pipeline_; }
    [[nodiscard]] const class FrameBuffer& swapChainFrameBuffer(const size_t i) const { return swap_chain_framebuffers_[i]; }

    [[nodiscard]] virtual const assets::Scene& getScene() const = 0;
    [[nodiscard]] virtual assets::UniformBufferObject getUniformBufferObject(VkExtent2D extent) const = 0;

    virtual void onDeviceSet();
    virtual void createSwapChain();
    virtual void deleteSwapChain();
    virtual void drawFrame();
    virtual void render(VkCommandBuffer command_buffer, uint32_t image_index);

    virtual void onKey(int key, int scancode, int action, int mods) {}
    virtual void onCursorPosition(double xpos, double ypos) {}
    virtual void onMouseButton(int button, int action, int mods) {}

    bool is_wire_frame_{};

private:

    void updateUniformBuffer(uint32_t image_index);
    void recreateSwapChain();

    const bool vsync_;
    std::unique_ptr<class Window> window_;
    std::unique_ptr<class Instance> instance_;
    std::unique_ptr<class Surface> surface_;
    std::unique_ptr<class Device> device_;
    std::unique_ptr<class SwapChain> swap_chain_;
    std::vector<assets::UniformBuffer> uniform_buffers_;
    std::unique_ptr<class DepthBuffer> depth_buffer_;
    std::unique_ptr<class GraphicsPipeline> graphics_pipeline_;
    std::vector<class FrameBuffer> swap_chain_framebuffers_;
    std::unique_ptr<class CommandPool> command_pool_;
    std::unique_ptr<class CommandBuffers> command_buffers_;
    std::vector<class Semaphore> image_available_semaphores_;
    std::vector<class Semaphore> render_finished_semaphores_;
    std::vector<class Fence> in_flight_fences_;

    size_t current_frame_{};
};

}
