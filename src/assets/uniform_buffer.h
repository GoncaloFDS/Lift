#pragma once

#include "core/glm.h"
#include "lights.h"
#include <memory>

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}  // namespace vulkan

namespace assets {
struct UniformBufferObject {
    Light light {};
    glm::mat4 modelView {};
    glm::mat4 projection {};
    glm::mat4 model_view_inverse {};
    glm::mat4 projection_inverse {};
    float aperture {};
    float focus_distance {};
    uint32_t total_number_of_samples {};
    uint32_t number_of_samples {};
    uint32_t number_of_bounces {};
    uint32_t seed {};
    uint32_t gamma_correction {};  // bool
    uint32_t tone_map {};  // bool
    float exposure {};
    uint32_t has_sky {};           // bool
    uint32_t frame {};
    uint32_t debug_normals {};  // bool
};

class UniformBuffer {
public:
    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer& operator=(const UniformBuffer&) = delete;
    UniformBuffer& operator=(UniformBuffer&&) = delete;

    explicit UniformBuffer(const vulkan::Device& device);
    UniformBuffer(UniformBuffer&& other) noexcept;
    ~UniformBuffer();

    [[nodiscard]] const vulkan::Buffer& buffer() const { return *buffer_; }

    void setValue(const UniformBufferObject& ubo);

private:
    std::unique_ptr<vulkan::Buffer> buffer_;
    std::unique_ptr<vulkan::DeviceMemory> memory_;
};

}  // namespace assets