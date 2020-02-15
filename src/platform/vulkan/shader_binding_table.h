#pragma once

#include "core/utilities.h"
#include <memory>
#include <vector>

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}  // namespace vulkan

namespace vulkan {
class DeviceProcedures;
class RayTracingPipeline;
class RayTracingProperties;

class ShaderBindingTable final {
public:
    struct Entry {
        uint32_t groupIndex;
        std::vector<unsigned char> inlineData;
    };
    ShaderBindingTable(const DeviceProcedures &device_procedures,
                       const RayTracingPipeline &ray_tracing_pipeline,
                       const RayTracingProperties &ray_tracing_properties,
                       const std::vector<Entry> &ray_gen_programs,
                       const std::vector<Entry> &miss_programs,
                       const std::vector<Entry> &hit_groups);

    ~ShaderBindingTable();

    [[nodiscard]] const class Buffer &buffer() const { return *buffer_; }

    [[nodiscard]] size_t rayGenOffset() const { return ray_gen_offset_; }
    [[nodiscard]] size_t missOffset() const { return miss_offset_; }
    [[nodiscard]] size_t hitGroupOffset() const { return hit_group_offset_; }

    [[nodiscard]] size_t rayGenEntrySize() const { return ray_gen_entry_size_; }
    [[nodiscard]] size_t missEntrySize() const { return miss_entry_size_; }
    [[nodiscard]] size_t hitGroupEntrySize() const { return hit_group_entry_size_; }

private:
    const size_t ray_gen_entry_size_;
    const size_t miss_entry_size_;
    const size_t hit_group_entry_size_;

    const size_t ray_gen_offset_;
    const size_t miss_offset_;
    const size_t hit_group_offset_;

    std::unique_ptr<class Buffer> buffer_;
    std::unique_ptr<DeviceMemory> buffer_memory_;
};

}  // namespace vulkan
