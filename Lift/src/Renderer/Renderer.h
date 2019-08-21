#pragma once
#include "RendererAPI.h"

#include "CudaBuffer.h"
#include "cuda/launch_parameters.cuh"
#include <optix_types.h>
#include <cuda_runtime.h>

class Camera;

namespace lift {
	struct TriangleMesh {
		void AddCube(const vec3& center, const vec3& size);

		std::vector<vec3> vertices;
		std::vector<ivec3> indices;
		vec3 color;
	};

	class Renderer {
	public:
		Renderer();
		void Init();
		void Render();
		void Resize(const ivec2& size);
		void DownloadPixels(uint32_t h_pixels[]);

		static void Submit(const std::shared_ptr<VertexArray>& vertex_array);
		static RendererAPI::API GetAPI() { return RendererAPI::GetAPI(); }

		void SetCamera(const Camera& camera);
		void AddModel(const TriangleMesh& model);
	protected:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------

		/*! helper function that initializes optix, and checks for errors */
		void InitOptix();
		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void CreateContext();
		/*! creates the module that contains all the programs we are going
		  to use. in this simple example, we use a single module from a
		  single .cu file, using a single embedded ptx string */
		void CreateModule();
		/*! does all setup for the raygen program(s) we are going to use */
		void CreateRaygenPrograms();
		/*! does all setup for the miss program(s) we are going to use */
		void CreateMissPrograms();
		/*! does all setup for the hitgroup program(s) we are going to use */
		void CreateHitgroupPrograms();
		/*! assembles the full pipeline of all programs */
		void CreatePipeline();
		/*! constructs the shader binding table */
		void BuildShaderBindingTables();

		OptixTraversableHandle BuildAccelerationStructure(const TriangleMesh& model);
	protected:
		/*! @{ CUDA device context and stream that optix pipeline will run
			on, as well as device properties for this device */
		CUcontext cuda_context_;
		CUstream stream_;
		cudaDeviceProp device_props_;
		/*! @} */

		//! the optix context that our pipeline will run in.
		OptixDeviceContext optix_context_;

		/*! @{ the pipeline we're building */
		OptixPipeline pipeline_;
		OptixPipelineCompileOptions pipeline_compile_options_;
		OptixPipelineLinkOptions pipeline_link_options_;
		/*! @} */

		/*! @{ the module that contains out device programs */
		OptixModule module_;
		OptixModuleCompileOptions module_compile_options_;
		/* @} */

		/*! vector of all our program(group)s, and the SBT built around
			them */
		std::vector<OptixProgramGroup> raygen_program_groups_;
		CudaBuffer raygen_program_groups_buffer_;
		std::vector<OptixProgramGroup> miss_program_groups_;
		CudaBuffer miss_program_groups_buffer_;
		std::vector<OptixProgramGroup> hit_program_groups_;
		CudaBuffer hit_program_groups_buffer_;
		OptixShaderBindingTable sbt_ = {};

		/*! @{ our launch parameters, on the host, and the buffer to store
			them on the device */
		LaunchParameters launch_params_;
		CudaBuffer launch_params_buffer_;
		/*! @} */

		CudaBuffer color_buffer_;

		TriangleMesh model_;
		CudaBuffer vertices_buffer_;
		CudaBuffer indices_buffer_;
		CudaBuffer acceleration_struct_buffer_;
	};
}
