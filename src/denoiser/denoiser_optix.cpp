#include "denoiser_optix.h"

#include "cuda.h"
#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include <core/log.h>
#include <platform/vulkan/single_time_commands.h>

OptixDeviceContext optix_device_context_;

static void contextLogCb(unsigned int level, const char* tag, const char* message, void* cbdata) {
    LF_INFO("[CUDA] {0}", message);
}

void DenoiserOptix::setup(vulkan::Device& device, uint32_t queue_index) {
    CUresult cuRes = cuInit(0);

    if (cuRes != CUDA_SUCCESS) {
        LF_ERROR("ERROR: initOptiX() cuInit() failed: {0}", cuRes);
        return;
    }

    CUdevice cuda_device = 0;
    cuRes = cuCtxCreate(&cuda_context_, CU_CTX_SCHED_SPIN, cuda_device);
    if (cuRes != CUDA_SUCCESS) {
        LF_ERROR("ERROR: initOptiX() cuCtxCreate() failed: {0}", cuRes);
        return;
    }

    // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
    cuRes = cuStreamCreate(&cuda_stream_, CU_STREAM_DEFAULT);
    if (cuRes != CUDA_SUCCESS) {
        LF_ERROR("ERROR: initOptiX() cuStreamCreate() failed: {0}", cuRes);
        return;
    }
    CUcontext cu_context;

    CUresult cu_result = cuCtxGetCurrent(&cu_context);
    if (cu_result != CUDA_SUCCESS) {
        LF_ERROR("Error querying current context: code -> {0}", cu_result);
    }
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(cu_context, nullptr, &optix_device_context_));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_device_context_, contextLogCb, nullptr, 4))

    OptixPixelFormat pixel_format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK(optixDenoiserCreate(optix_device_context_,
                                    OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR,
                                    &denoiser_options_,
                                    &denoiser_));
    LF_INFO("Initialized Optix Denoiser");

    vk_allocator_.init(device.handle(), device.physicalDevice());
}

void DenoiserOptix::denoiseImage(vulkan::Device& device,
                                 VkCommandBuffer& command_buffer,
                                 vulkan::CommandPool& command_pool,
                                 vulkan::Image& in_image,
                                 vulkan::Image& out_image) {
    auto extent = in_image.extent();
    if (image_size_.height != extent.height || image_size_.width != extent.width) {
        image_size_.width = extent.width;
        image_size_.height = extent.height;
        allocateBuffers(device);
    }

    in_image.transitionImageLayout(command_pool, vk::ImageLayout::eTransferSrcOptimal);
    in_image.copyToBuffer(command_pool, pixel_buffer_in_.buf_vk.buffer);
    in_image.transitionImageLayout(command_pool, vk::ImageLayout::eGeneral);

    uint32_t rowStrideInBytes = image_size_.width * sizeof(float4);
    auto size_of_pixel = static_cast<uint32_t>(sizeof(float4));
    OptixPixelFormat pixel_format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D input_layer;
    input_layer.data = (CUdeviceptr) pixel_buffer_in_.cuda_ptr;
    input_layer.width = image_size_.width;
    input_layer.height = image_size_.height;
    input_layer.rowStrideInBytes = rowStrideInBytes;
    input_layer.pixelStrideInBytes = size_of_pixel;
    input_layer.format = pixel_format;

    OptixImage2D output_layer;
    output_layer.data = (CUdeviceptr) pixel_buffer_out_.cuda_ptr;
    output_layer.width = image_size_.width;
    output_layer.height = image_size_.height;
    output_layer.rowStrideInBytes = rowStrideInBytes;
    output_layer.pixelStrideInBytes = size_of_pixel;
    output_layer.format = pixel_format;

    CUstream stream = nullptr;
    OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser_,
                                              stream,
                                              &input_layer,
                                              p_intensity_,
                                              p_scratch_,
                                              denoiser_sizes_.withoutOverlapScratchSizeInBytes));

    OptixDenoiserParams denoiser_params {};
    denoiser_params.denoiseAlpha = false;
    denoiser_params.hdrIntensity = p_intensity_;
    denoiser_params.blendFactor = 0.0f;

    OptixDenoiserGuideLayer guide_layer {};
    guide_layer.albedo = input_layer;

    OptixDenoiserLayer denoiser_layer {};
    denoiser_layer.input = input_layer;
    denoiser_layer.output = output_layer;
    if (previous_output_.data) {
        denoiser_layer.previousOutput = previous_output_;
    }


    OPTIX_CHECK(optixDenoiserInvoke(denoiser_,
                                    stream,
                                    &denoiser_params,
                                    p_state_,
                                    denoiser_sizes_.stateSizeInBytes,
                                    &guide_layer,
                                    &denoiser_layer,
                                    1,
                                    0,
                                    0,
                                    p_scratch_,
                                    denoiser_sizes_.withoutOverlapScratchSizeInBytes));

    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    out_image.transitionImageLayout(command_pool, vk::ImageLayout::eTransferDstOptimal);
    out_image.copyFromBuffer(command_pool, pixel_buffer_out_.buf_vk.buffer);
    out_image.transitionImageLayout(command_pool, vk::ImageLayout::eGeneral);

    previous_output_ = output_layer;
}

void DenoiserOptix::allocateBuffers(vulkan::Device& device) {
    destroy();

    vk::DeviceSize buffer_size = image_size_.width * image_size_.height * sizeof(float) * 4;

    vk::BufferUsageFlags usage_flags {vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst |
                                      vk::BufferUsageFlagBits::eTransferSrc};

    pixel_buffer_in_.buf_vk =
        vk_allocator_.createBuffer(static_cast<VkDeviceSize>(buffer_size),
                                   static_cast<VkBufferUsageFlags>(usage_flags),
                                   static_cast<VkMemoryPropertyFlags>(vk::MemoryPropertyFlagBits::eDeviceLocal));
    pixel_buffer_out_.buf_vk =
        vk_allocator_.createBuffer(static_cast<VkDeviceSize>(buffer_size),
                                   static_cast<VkBufferUsageFlags>(usage_flags),
                                   static_cast<VkMemoryPropertyFlags>(vk::MemoryPropertyFlagBits::eDeviceLocal));

    createBufferCuda(device, pixel_buffer_in_);
    createBufferCuda(device, pixel_buffer_out_);

    OPTIX_CHECK(
        optixDenoiserComputeMemoryResources(denoiser_, image_size_.width, image_size_.height, &denoiser_sizes_));

    cudaMalloc((void**) &p_state_, denoiser_sizes_.stateSizeInBytes);
    cudaMalloc((void**) &p_scratch_, denoiser_sizes_.withoutOverlapScratchSizeInBytes);
    cudaMalloc((void**) &p_intensity_, sizeof(float));
    cudaMalloc((void**) &p_min_rgb_, 4 * sizeof(float));

    CUstream stream = cuda_stream_;
    OPTIX_CHECK(optixDenoiserSetup(denoiser_,
                                   stream,
                                   image_size_.width,
                                   image_size_.height,
                                   p_state_,
                                   denoiser_sizes_.stateSizeInBytes,
                                   p_scratch_,
                                   denoiser_sizes_.withoutOverlapScratchSizeInBytes));
}

void DenoiserOptix::createBufferCuda(vulkan::Device& device, CudaBuffer& cuda_buffer) {
    vk::Device vkDevice {device.handle()};

    cuda_buffer.handle = vkDevice.getMemoryWin32HandleKHR(
        {cuda_buffer.buf_vk.allocation, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
    auto req = vkDevice.getBufferMemoryRequirements(cuda_buffer.buf_vk.buffer);

    cudaExternalMemoryHandleDesc cuda_ext_memory_handle_desc {};
    cuda_ext_memory_handle_desc.size = req.size;
    cuda_ext_memory_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuda_ext_memory_handle_desc.handle.win32.handle = cuda_buffer.handle;

    cudaExternalMemory_t cuda_ext_vertex_buffer {};
    CUDA_CHECK(cudaImportExternalMemory(&cuda_ext_vertex_buffer, &cuda_ext_memory_handle_desc));

    cudaExternalMemoryBufferDesc cuda_ext_buffer_desc {};
    cuda_ext_buffer_desc.offset = 0;
    cuda_ext_buffer_desc.size = req.size;
    cuda_ext_buffer_desc.flags = 0;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cuda_buffer.cuda_ptr, cuda_ext_vertex_buffer, &cuda_ext_buffer_desc));
}

void DenoiserOptix::destroy() {
    pixel_buffer_in_.destroy(vk_allocator_);
    pixel_buffer_out_.destroy(vk_allocator_);

    if (p_state_ != 0) {
        CUDA_CHECK(cudaFree((void*) p_state_));
    }
    if (p_scratch_ != 0) {
        CUDA_CHECK(cudaFree((void*) p_scratch_));
    }
    if (p_intensity_ != 0) {
        CUDA_CHECK(cudaFree((void*) p_intensity_));
    }
    if (p_min_rgb_ != 0) {
        CUDA_CHECK(cudaFree((void*) p_min_rgb_));
    }
}
