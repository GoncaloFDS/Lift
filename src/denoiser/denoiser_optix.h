#pragma once

#include "optix_types.h"
#include "vulkan/buffer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <platform/nvvkpp/allocator_dedicated_vkpp.hpp>
#include <platform/vulkan/image.h>
#include <vulkan/device.h>

class DenoiserOptix {
public:
    struct CudaBuffer {
        //        std::unique_ptr<vulkan::Buffer> buffer_vk_;
        nvvkpp::BufferDedicated buf_vk;

        HANDLE handle = nullptr;
        void* cuda_ptr = nullptr;

        void destroy(nvvkpp::AllocatorVkExport& alloc) {
            alloc.destroy(buf_vk);
            CloseHandle(handle);
        }
    };

    DenoiserOptix();

    void setup(vulkan::Device&, uint32_t queue_index);
    int initOptix();
    void denoiseImage(vulkan::Device& device,
                      VkCommandBuffer& command_buffer,
                      vulkan::CommandPool& command_pool,
                      vulkan::Image& in_image,
                      vulkan::Image& out_image);
    void destroy();
    void createBufferCuda(vulkan::Device& device, CudaBuffer& cuda_buffer);

    int denoise_mode {1};
    int start_denoiser_frame {5};

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

    nvvkpp::AllocatorVkExport vk_allocator_;

    VkExtent2D image_size_ {};
    CudaBuffer pixel_buffer_in_;
    CudaBuffer pixel_buffer_out_;
};
