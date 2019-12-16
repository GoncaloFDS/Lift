#pragma once

#include "platform/vulkan/Application.h"
#include "ray_tracing_properties.h"

namespace vulkan {
class CommandBuffers;
class Buffer;
class DeviceMemory;
class Image;
class ImageView;
}

namespace vulkan::ray_tracing {
class Application : public vulkan::Application {
public:
protected:

    Application(const WindowProperties& window_properties, bool vsync, bool enable_validation_layers);
    ~Application();

    void onDeviceSet() override;
    void createAccelerationStructures();
    void deleteAccelerationStructures();
    void createSwapChain() override;
    void deleteSwapChain() override;
    void render(VkCommandBuffer command_buffer, uint32_t image_index) override;

private:

    void createBottomLevelStructures(VkCommandBuffer command_buffer);
    void createTopLevelStructures(VkCommandBuffer command_buffer);
    void createOutputImage();

    std::unique_ptr<class RayTracingProperties> properties_;
    std::unique_ptr<class DeviceProcedures> device_procedures_;

    std::vector<class BottomLevelAccelerationStructure> bottom_as_;
    std::unique_ptr<Buffer> bottom_buffer_;
    std::unique_ptr<DeviceMemory> bottom_buffer_memory_;
    std::unique_ptr<Buffer> bottom_scratch_buffer_;
    std::unique_ptr<DeviceMemory> bottom_scratch_buffer_memory_;
    std::vector<class TopLevelAccelerationStructure> top_as_;
    std::unique_ptr<Buffer> top_buffer_;
    std::unique_ptr<DeviceMemory> top_buffer_memory_;
    std::unique_ptr<Buffer> top_scratch_buffer_;
    std::unique_ptr<DeviceMemory> top_scratch_buffer_memory_;
    std::unique_ptr<Buffer> instances_buffer_;
    std::unique_ptr<DeviceMemory> instances_buffer_memory_;

    std::unique_ptr<Image> accumulation_image_;
    std::unique_ptr<DeviceMemory> accumulation_image_memory_;
    std::unique_ptr<ImageView> accumulation_image_view_;

    std::unique_ptr<Image> output_image_;
    std::unique_ptr<DeviceMemory> output_image_memory_;
    std::unique_ptr<ImageView> output_image_view_;

    std::unique_ptr<class RayTracingPipeline> ray_tracing_pipeline_;
    std::unique_ptr<class ShaderBindingTable> shader_binding_table_;
};

}
