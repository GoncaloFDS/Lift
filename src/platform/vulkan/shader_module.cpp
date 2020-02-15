#include "shader_module.h"
#include "device.h"
#include <fstream>

namespace vulkan {

ShaderModule::ShaderModule(const class Device &device, const std::string &filename) :
    ShaderModule(device, readFile(filename)) {
}

ShaderModule::ShaderModule(const class Device &device, const std::vector<char> &code) : device_(device) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

    vulkanCheck(vkCreateShaderModule(device.handle(), &create_info, nullptr, &shader_module_), "create shader module");
}

ShaderModule::~ShaderModule() {
    if (shader_module_ != nullptr) {
        vkDestroyShaderModule(device_.handle(), shader_module_, nullptr);
        shader_module_ = nullptr;
    }
}

VkPipelineShaderStageCreateInfo ShaderModule::createShaderStage(VkShaderStageFlagBits stage) const {
    VkPipelineShaderStageCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    create_info.stage = stage;
    create_info.module = shader_module_;
    create_info.pName = "main";

    return create_info;
}

std::vector<char> ShaderModule::readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        //		Throw(std::runtime_error("failed to open file '" + filename + "'"));
    }

    const auto file_size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
}

}  // namespace vulkan
