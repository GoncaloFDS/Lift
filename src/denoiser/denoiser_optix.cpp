#include <pch.h>
#include "denoiser_optix.h"
#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

OptixDeviceContext optix_device_context_;

static void context_log_cb(unsigned int level, const char *tag, const char* message, void* cbdata) {
	LF_WARN("[CUDA] {0}", message);
}

DenoiserOptix::DenoiserOptix() {

}

void DenoiserOptix::setup(vulkan::Device device, uint32_t queue_index) {

}

int DenoiserOptix::initOptix() {
	cudaFree(nullptr);

	CUcontext cu_context;
	CUresult cu_result = cuCtxGetCurrent(&cu_context);
	if (cu_result != CUDA_SUCCESS) {
		LF_ERROR("Error querying current context: code -> {0}", cu_result);
	}
	OPTIX_CHECK(optixInit());
	OPTIX_CHECK(optixDeviceContextCreate(cu_context, nullptr, &optix_device_context_));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_device_context_, context_log_cb, nullptr, 4))

	OptixPixelFormat pixel_format = OPTIX_PIXEL_FORMAT_FLOAT4;

	denoiser_options_.inputKind = OPTIX_DENOISER_INPUT_RGB;
	denoiser_options_.pixelFormat = pixel_format;
	OPTIX_CHECK(optixDenoiserCreate(optix_device_context_, &denoiser_options_, &denoiser_));
	OPTIX_CHECK(optixDenoiserSetModel(denoiser_, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
	LF_INFO("Initialized Optix Denoiser");
	return 1;
}
