#pragma once


#include "optix_types.h"
#include "vulkan/buffer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvvk/allocator_dedicated_vk.hpp>
#include <vulkan/image.h>
#include <vulkan/device.h>

struct CudaBuffer {
    //        std::unique_ptr<vulkan::Buffer> buffer_vk_;
    nvvk::BufferDedicated buf_vk;

    HANDLE handle = nullptr;
    void* cuda_ptr = nullptr;

    void destroy(nvvk::AllocatorVkExport& alloc) {
        alloc.destroy(buf_vk);
        CloseHandle(handle);
    }
};

class DenoiserOptix {
public:

    void setup(vulkan::Device&, uint32_t queue_index);
    void denoiseImage(vulkan::Device& device,
                      VkCommandBuffer& command_buffer,
                      vulkan::CommandPool& command_pool,
                      vulkan::Image& in_image,
                      vulkan::Image& out_image);
    void destroy();
    void createBufferCuda(vulkan::Device& device, CudaBuffer& cuda_buffer);


private:
    void allocateBuffers(vulkan::Device& device);

    OptixDenoiser denoiser_ {};
    OptixDenoiserOptions denoiser_options_ {};
    OptixDenoiserSizes denoiser_sizes_ {};
    CUdeviceptr p_state_ {0};
    CUdeviceptr p_scratch_ {0};
    CUdeviceptr p_intensity_ {0};
    CUdeviceptr p_min_rgb_ {0};

    uint32_t queue_index_ {};

    nvvk::AllocatorVkExport vk_allocator_;

    VkExtent2D image_size_ {};
    CudaBuffer pixel_buffer_in_;
    CudaBuffer pixel_buffer_out_;
};
