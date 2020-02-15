#pragma once

#include "core/utilities.h"
#include <string>
#include <vector>

namespace vulkan {
class Device;

class ShaderModule final {
public:
    ShaderModule(const Device &device, const std::string &filename);
    ShaderModule(const Device &device, const std::vector<char> &code);
    ~ShaderModule();

    [[nodiscard]] VkShaderModule handle() const { return shader_module_; }
    [[nodiscard]] const Device &device() const { return device_; }

    [[nodiscard]] VkPipelineShaderStageCreateInfo createShaderStage(VkShaderStageFlagBits stage) const;

private:
    static std::vector<char> readFile(const std::string &filename);

    const class Device &device_;
    VkShaderModule shader_module_ {};
};

}  // namespace vulkan
