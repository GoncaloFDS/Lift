#include "sampler.h"
#include "device.h"

namespace vulkan {

Sampler::Sampler(const class Device &device, const SamplerConfig &config) : device_(device) {
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = config.magFilter;
  samplerInfo.minFilter = config.minFilter;
  samplerInfo.addressModeU = config.addressModeU;
  samplerInfo.addressModeV = config.addressModeV;
  samplerInfo.addressModeW = config.addressModeW;
  samplerInfo.anisotropyEnable = config.anisotropyEnable;
  samplerInfo.maxAnisotropy = config.maxAnisotropy;
  samplerInfo.borderColor = config.borderColor;
  samplerInfo.unnormalizedCoordinates = config.unnormalizedCoordinates;
  samplerInfo.compareEnable = config.compareEnable;
  samplerInfo.compareOp = config.compareOp;
  samplerInfo.mipmapMode = config.mipmapMode;
  samplerInfo.mipLodBias = config.mipLodBias;
  samplerInfo.minLod = config.minLod;
  samplerInfo.maxLod = config.maxLod;

  if (vkCreateSampler(device.handle(), &samplerInfo, nullptr, &sampler_) != VK_SUCCESS) {
    //		Throw(std::runtime_error("failed to create sampler"));
  }
}

Sampler::~Sampler() {
  if (sampler_ != nullptr) {
    vkDestroySampler(device_.handle(), sampler_, nullptr);
    sampler_ = nullptr;
  }
}

}  // namespace vulkan
