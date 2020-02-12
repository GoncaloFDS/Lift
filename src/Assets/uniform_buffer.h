#pragma once

#include "core/glm.h"
#include <memory>

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}  // namespace vulkan

namespace assets {
class UniformBufferObject {
  public:
  glm::mat4 modelView;
  glm::mat4 projection;
  glm::mat4 modelViewInverse;
  glm::mat4 projectionInverse;
  float aperture;
  float focusDistance;
  uint32_t totalNumberOfSamples;
  uint32_t numberOfSamples;
  uint32_t numberOfBounces;
  uint32_t randomSeed;
  uint32_t gammaCorrection;  // bool
  uint32_t hasSky;           // bool
};

class UniformBuffer {
  public:
  UniformBuffer(const UniformBuffer &) = delete;
  UniformBuffer &operator=(const UniformBuffer &) = delete;
  UniformBuffer &operator=(UniformBuffer &&) = delete;

  explicit UniformBuffer(const vulkan::Device &device);
  UniformBuffer(UniformBuffer &&other) noexcept;
  ~UniformBuffer();

  [[nodiscard]] const vulkan::Buffer &buffer() const { return *buffer_; }

  void setValue(const UniformBufferObject &ubo);

  private:
  std::unique_ptr<vulkan::Buffer> buffer_;
  std::unique_ptr<vulkan::DeviceMemory> memory_;
};

}  // namespace assets