#pragma once

#include "core/glm.h"
#include <memory>

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}

namespace assets {
class UniformBufferObject {
public:

    glm::mat4 ModelView;
    glm::mat4 Projection;
    glm::mat4 ModelViewInverse;
    glm::mat4 ProjectionInverse;
    float Aperture;
    float FocusDistance;
    uint32_t TotalNumberOfSamples;
    uint32_t NumberOfSamples;
    uint32_t NumberOfBounces;
    uint32_t RandomSeed;
    uint32_t GammaCorrection; // bool
    uint32_t HasSky; // bool
};

class UniformBuffer {
public:

    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer& operator=(const UniformBuffer&) = delete;
    UniformBuffer& operator=(UniformBuffer&&) = delete;

    explicit UniformBuffer(const vulkan::Device& device);
    UniformBuffer(UniformBuffer&& other) noexcept;
    ~UniformBuffer();

    const vulkan::Buffer& Buffer() const { return *buffer_; }

    void SetValue(const UniformBufferObject& ubo);

private:

    std::unique_ptr<vulkan::Buffer> buffer_;
    std::unique_ptr<vulkan::DeviceMemory> memory_;
};

}