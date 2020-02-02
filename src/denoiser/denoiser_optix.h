#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vulkan/device.h>
#include "optix_types.h"
#include "vulkan/buffer.h"

class DenoiserOptix {
public:

	struct CudaBuffer {
		vulkan::Buffer buffer_vk_;

		void* handle = nullptr;

	};

	DenoiserOptix();
	void setup(const vulkan::Device, uint32_t queue_index);
	int initOptix();
	void denoiseImage(const VkDescriptorImageInfo& image_in, VkDescriptorImageInfo* image_out, const VkExtent2D& image_size);
	void destroy();
	void createBufferCuda(CudaBuffer& cuda_buffer);
	void importMemory();
	void uiSetup();

	int denoise_mode {1};
	int start_denoiser_frame {5};

private:
	void allocateBuffers();
	void bufferToImage(const vulkan::Buffer& pixel_buffer_out, VkDescriptorImageInfo* image_out);
	void imageToBuffer(const VkDescriptorImageInfo& image_in, const vulkan::Buffer& pixel_buffer_in);

	OptixDenoiser denoiser_{};
	OptixDenoiserOptions denoiser_options_{};
	OptixDenoiserSizes denoiser_sizes_{};
	CUdeviceptr p_state_{0};
	CUdeviceptr p_scratch_{0};
	CUdeviceptr p_intensity_{0};
	CUdeviceptr p_min_rgb_{0};

	uint32_t queue_index_{};

	VkExtent2D image_size_{};
//	CudaBuffer pixel_buffer_in_;
//	CudaBuffer pixel_buffer_out_;
};

