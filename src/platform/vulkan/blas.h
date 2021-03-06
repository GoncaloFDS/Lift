#pragma once

#include "acceleration_structure.h"
#include "bottom_level_geometry.h"

namespace assets {
class Procedural;
class Scene;
}  // namespace assets

namespace vulkan {

class BottomLevelAccelerationStructure final : public AccelerationStructure {
public:
    BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(BottomLevelAccelerationStructure&&) = delete;

    BottomLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                     const BottomLevelGeometry& geometries);
    BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept;
    ~BottomLevelAccelerationStructure();

    void generate(VkCommandBuffer command_buffer,
                  Buffer& scratch_buffer,
                  VkDeviceSize scratch_offset,
                  Buffer& result_buffer,
                  VkDeviceSize result_offset);

private:
   BottomLevelGeometry geometries_;
};

}  // namespace vulkan
