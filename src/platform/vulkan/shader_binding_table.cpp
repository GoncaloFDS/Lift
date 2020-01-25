#include "shader_binding_table.h"
#include "device_procedures.h"
#include "ray_tracing_pipeline.h"
#include "ray_tracing_properties.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/device.h"
#include "platform/vulkan/device_memory.h"
#include <algorithm>
#include <cstring>
#include <memory>

namespace vulkan {

namespace {
size_t RoundUp(size_t size, size_t powerOf2Alignment) {
    return (size + powerOf2Alignment - 1) & ~(powerOf2Alignment - 1);
}

size_t GetEntrySize(const RayTracingProperties& rayTracingProperties,
                    const std::vector<ShaderBindingTable::Entry>& entries) {
    // Find the maximum number of parameters used by a single entry
    size_t maxArgs = 0;

    for (const auto& entry : entries) {
        maxArgs = std::max(maxArgs, entry.InlineData.size());
    }

    // A SBT entry is made of a program ID and a set of 4-byte parameters (offsets or push constants)
    // and must be 16-bytes-aligned.
    return RoundUp(rayTracingProperties.shaderGroupHandleSize() + maxArgs, 16);
}

size_t CopyShaderData(
    uint8_t* const dst,
    const RayTracingProperties& rayTracingProperties,
    const std::vector<ShaderBindingTable::Entry>& entries,
    const size_t entrySize,
    const uint8_t* const shaderHandleStorage) {
    const auto handleSize = rayTracingProperties.shaderGroupHandleSize();

    uint8_t* pDst = dst;

    for (const auto& entry : entries) {
        // Copy the shader identifier that was previously obtained with vkGetRayTracingShaderGroupHandlesNV.
        std::memcpy(pDst, shaderHandleStorage + entry.GroupIndex * handleSize, handleSize);
        std::memcpy(pDst + handleSize, entry.InlineData.data(), entry.InlineData.size());

        pDst += entrySize;
    }

    return entries.size() * entrySize;
}

}

ShaderBindingTable::ShaderBindingTable(
    const DeviceProcedures& device_procedures,
    const RayTracingPipeline& ray_tracing_pipeline,
    const RayTracingProperties& ray_tracing_properties,
    const std::vector<Entry>& ray_gen_programs,
    const std::vector<Entry>& miss_programs,
    const std::vector<Entry>& hit_groups) :
    ray_gen_entry_size_(GetEntrySize(ray_tracing_properties, ray_gen_programs)),
    miss_entry_size_(GetEntrySize(ray_tracing_properties, miss_programs)),
    hit_group_entry_size_(GetEntrySize(ray_tracing_properties, hit_groups)),
    ray_gen_offset_(0),
    miss_offset_(ray_gen_programs.size() * ray_gen_entry_size_),
    hit_group_offset_(miss_offset_ + miss_programs.size() * miss_entry_size_) {
    // Compute the size of the table.
    const size_t sbtSize =
        ray_gen_programs.size() * ray_gen_entry_size_ +
            miss_programs.size() * miss_entry_size_ +
            hit_groups.size() * hit_group_entry_size_;

    // Allocate buffer & memory.
    const auto& device = ray_tracing_properties.device();

    buffer_ = std::make_unique<class Buffer>(device, sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    buffer_memory_ = std::make_unique<DeviceMemory>(buffer_->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));

    // Generate the table.
    const uint32_t handleSize = ray_tracing_properties.shaderGroupHandleSize();
    const size_t groupCount = ray_gen_programs.size() + miss_programs.size() + hit_groups.size();
    std::vector<uint8_t> shaderHandleStorage(groupCount * handleSize);

    vulkanCheck(device_procedures.vkGetRayTracingShaderGroupHandlesNV(
        device.handle(),
        ray_tracing_pipeline.handle(),
        0, static_cast<uint32_t>(groupCount),
        shaderHandleStorage.size(),
        shaderHandleStorage.data()),
                "get ray tracing shader group handles");

    // Copy the shader identifiers followed by their resource pointers or root constants:
    // first the ray generation, then the miss shaders, and finally the set of hit groups.
    auto pData = static_cast<uint8_t*>(buffer_memory_->map(0, sbtSize));

    pData += CopyShaderData(pData, ray_tracing_properties, ray_gen_programs, ray_gen_entry_size_, shaderHandleStorage.data());
    pData += CopyShaderData(pData, ray_tracing_properties, miss_programs, miss_entry_size_, shaderHandleStorage.data());
    CopyShaderData(pData, ray_tracing_properties, hit_groups, hit_group_entry_size_, shaderHandleStorage.data());

    buffer_memory_->unmap();
}

ShaderBindingTable::~ShaderBindingTable() {
    buffer_.reset();
    buffer_memory_.reset();
}

}
