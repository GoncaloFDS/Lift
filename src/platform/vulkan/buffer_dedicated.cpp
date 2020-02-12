//#include "buffer_dedicated.h"
//
// BufferDedicated createBuffer(vk::Device& device,
//                             const vk::MemoryPropertyFlagBits mem_usage,
//                             const vk::BufferCreateInfo& info) {
//
//    BufferDedicated result_buffer;
//
//    result_buffer.buffer = device.createBuffer(info);
//
//    auto r = device.getBufferMemoryRequirements2
//        <vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(result_buffer.buffer);
//    // Find Memory Requirements
//    vk::MemoryRequirements2 requirements_2 = r.get<vk::MemoryRequirements2>();
//    vk::MemoryDedicatedRequirements d = r.get<vk::MemoryDedicatedRequirements>();
//    vk::MemoryRequirements& memory_requirements = requirements_2.memoryRequirements;
//
//    // Allocate Memory
//    vk::MemoryAllocateInfo memory_info;
//    memory_info.setAllocationSize(memory_requirements.size);
//    memory_info.setMemoryTypeIndex(getMemoryType(memory_requirements.memoryTypeBits, mem_usage));
//    result_buffer.allocation = AllocateMemory(memory_info);
//    checkMemory(result_buffer.allocation);
//
//    // Bind Buffers
//    device.bindBufferMemory(result_buffer.buffer, result_buffer.allocation, 0);
//
//    return result_buffer;
//}
//
// uint32_t getMemoryType(uint32_t type_bits)