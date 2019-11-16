#include "pch.h"
#include "renderer.h"

#include <optix.h>
#include <optix_stubs.h>
#include <core/profiler.h>
#include <platform/opengl/opengl_display.h>
#include <GLFW/glfw3.h>
#include <cuda/launch_parameters.h>
#include <cuda/math_constructors.h>
#include <cuda/vec_math.h>
#include <array>
#include "scene/scene.h"
#include "cuda_buffer.h"
#include "record.h"

extern "C" char embedded_ptx_code[];

void lift::Renderer::init(CudaOutputBufferType output_buffer_type, ivec2 frame_size) {
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_params_ ), sizeof(LaunchParameters)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &launch_parameters_.accum_buffer ),
						  frame_size.x * frame_size.y * sizeof(float4)
	));
	launch_parameters_.frame_buffer = nullptr;
	launch_parameters_.subframe_index = 0u;
	launch_parameters_.samples_per_launch = 1;
	setClearColor(vec3(0.1f));
	createOutputBuffer(output_buffer_type, frame_size);
}

void lift::Renderer::launchSubframe(const Scene& scene, const ivec2& size) {
	if (size.x == 0 || size.y == 0)
		return;
	Profiler profiler(Profiler::Id::Render);
	uchar4* result_buffer_data = output_buffer_->map();
	launch_parameters_.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params_),
							   &launch_parameters_,
							   sizeof(LaunchParameters),
							   cudaMemcpyHostToDevice,
							   nullptr));

	OPTIX_CHECK(optixLaunch(
		this->pipeline(),
		nullptr,
		reinterpret_cast<CUdeviceptr>( d_params_ ),
		sizeof(LaunchParameters),
		this->sbt(),
		size.x,
		size.y,
		1));

	output_buffer_->unmap();
	CUDA_SYNC_CHECK();
}

void lift::Renderer::displaySubframe(OpenGLDisplay& gl_display, void* window) {
	Profiler profiler(Profiler::Id::Display);
	int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;   //
	glfwGetFramebufferSize(static_cast<GLFWwindow*>(window), &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		ivec2(output_buffer_->width(), output_buffer_->height()),
		ivec2(framebuf_res_x, framebuf_res_y),
		output_buffer_->getPixelBufferObject()
	);
}

void lift::Renderer::createOutputBuffer(CudaOutputBufferType type, ivec2 frame_size) {
	output_buffer_ = std::make_unique<CudaOutputBuffer<uchar4>>(type, frame_size.x, frame_size.y);

}

void lift::Renderer::allocLights(Scene& scene) {
	auto& lights = scene.lights();
	launch_parameters_.lights.count = static_cast<uint32_t>(lights.size());
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&launch_parameters_.lights.data),
		lights.size() * sizeof(Lights::ParallelogramLight)

	));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>( launch_parameters_.lights.data ),
		lights.data(),
		lights.size() * sizeof(Lights::ParallelogramLight),
		cudaMemcpyHostToDevice
	));
}

void lift::Renderer::updateLaunchParameters(Scene scene) {
	auto camera = scene.camera();

	launch_parameters_.camera.eye = makeFloat3(camera->eye());
	launch_parameters_.camera.u = makeFloat3(camera->vectorU());
	launch_parameters_.camera.v = makeFloat3(camera->vectorV());
	launch_parameters_.camera.w = makeFloat3(camera->vectorW());
	launch_parameters_.handle = this->traversableHandle();
}

void lift::Renderer::onResize(int32_t width, int32_t height) {
	resizeOutputBuffer(width, height);
	resizeAccumulationButter(width, height);
}

void lift::Renderer::resizeOutputBuffer(int32_t width, int32_t height) {
	output_buffer_->resize(width, height);
}

void lift::Renderer::resizeAccumulationButter(int32_t width, int32_t height) {
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>( launch_parameters_.accum_buffer )));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &launch_parameters_.accum_buffer ),
						  width * height * sizeof(float4)
	));
}

void lift::Renderer::setClearColor(const vec3& color) {
	clear_color_ = color;

	//launch_parameters_.miss_color = makeFloat3(clear_color_);
}

auto lift::Renderer::clearColor() -> vec3 {
	return clear_color_;
}

void lift::Renderer::resetFrame() {
	launch_parameters_.subframe_index = 0u;
}

void lift::Renderer::initOptix(const lift::Scene& scene) {

	createContext();
	buildMeshAccels(scene);
	buildInstanceAccel(scene, k_RayTypeCount);
	createPtxModule();
	createProgramGroups();
	createPipeline();
	createSbt(scene);

}

void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
	LF_INFO("[OptiX Log] {0}", message);
}

void lift::Renderer::createContext() {
	// Initialize CUDA
	CUDA_CHECK(cudaFree(nullptr));

	CUcontext cu_context = nullptr; // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &contextLogCallback;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cu_context, &options, &context_));
}

void lift::Renderer::buildMeshAccels(const Scene& scene) {
	// see explanation above
	constexpr double initial_compaction_ratio = 0.5;

	// It is assumed that trace is called later when the GASes are still in memory.
	// We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
	// Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
	constexpr size_t additional_available_memory = 250 * 1024 * 1024;

	//////////////////////////////////////////////////////////////////////////

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	struct GasInfo {
		std::vector<OptixBuildInput> build_inputs;
		OptixAccelBufferSizes gas_buffer_sizes;
		std::shared_ptr<Mesh> mesh;
	};
	std::multimap<size_t, GasInfo> gases;
	size_t total_temp_output_size = 0;
	/*const*/
	uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

	for (auto& mesh : scene.meshes()) {
		const size_t num_sub_meshes = mesh->indices.size();
		std::vector<OptixBuildInput> build_inputs(num_sub_meshes);

		LF_ASSERT(mesh->positions.size() == num_sub_meshes &&
			mesh->normals.size() == num_sub_meshes &&
			mesh->tex_coords.size() == num_sub_meshes, "Mesh components size mismatch");

		for (size_t i = 0; i < num_sub_meshes; ++i) {
			OptixBuildInput& triangle_input = build_inputs[i];
			memset(&triangle_input, 0, sizeof(OptixBuildInput));
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.vertexStrideInBytes =
				mesh->positions[i].byte_stride ? mesh->positions[i].byte_stride : sizeof(float3),
				triangle_input.triangleArray.numVertices = mesh->positions[i].count;
			triangle_input.triangleArray.vertexBuffers = &(mesh->positions[i].data);
			triangle_input.triangleArray.indexFormat =
				mesh->indices[i].elmt_byte_size == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
													 : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangle_input.triangleArray.indexStrideInBytes =
				mesh->indices[i].byte_stride ? mesh->indices[i].byte_stride : mesh->indices[i].elmt_byte_size * 3;
			triangle_input.triangleArray.numIndexTriplets = mesh->indices[i].count / 3;
			triangle_input.triangleArray.indexBuffer = mesh->indices[i].data;
			triangle_input.triangleArray.flags = &triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;
		}

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accel_options, build_inputs.data(),
												 static_cast<unsigned int>( num_sub_meshes ), &gas_buffer_sizes));

		total_temp_output_size += gas_buffer_sizes.outputSizeInBytes;
		GasInfo g = {std::move(build_inputs), gas_buffer_sizes, mesh};
		gases.emplace(gas_buffer_sizes.outputSizeInBytes, g);
	}

	size_t total_temp_output_processed_size = 0;
	size_t used_compacted_output_size = 0;
	double compaction_ratio = initial_compaction_ratio;

	CudaBuffer<char> d_temp;
	CudaBuffer<char> d_temp_output;
	CudaBuffer<size_t> d_temp_compacted_sizes;

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

	while (!gases.empty()) {
		// The estimated total output size that we end up with when using compaction.
		// It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
		// Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
		auto remaining_estimated_total_output_size =
			static_cast<size_t>((total_temp_output_size - total_temp_output_processed_size) * compaction_ratio);
		auto available_mem_pool_size = remaining_estimated_total_output_size + additional_available_memory;
		// We need to fit the following things into availableMemPoolSize:
		// - temporary buffer for building a GAS (only during build, can be cleared before compaction)
		// - build output buffer of a GAS
		// - size (actual number) of a compacted GAS as output of a build
		// - compacted GAS

		size_t batch_nga_ses = 0;
		size_t batch_build_output_requirement = 0;
		size_t batch_build_max_temp_requirement = 0;
		size_t batch_build_compacted_requirement = 0;
		for (auto it = gases.rbegin(); it != gases.rend(); it++) {
			batch_build_output_requirement += it->second.gas_buffer_sizes.outputSizeInBytes;
			batch_build_compacted_requirement += (size_t)(it->second.gas_buffer_sizes.outputSizeInBytes *
				compaction_ratio);
			// roughly account for the storage of the compacted size, although that goes into a separate buffer
			batch_build_output_requirement += 8ull;
			// make sure that all further output pointers are 256 byte aligned
			batch_build_output_requirement = roundUp<size_t>(batch_build_output_requirement, 256ull);
			// temp buffer is shared for all builds in the batch
			batch_build_max_temp_requirement = std::max(batch_build_max_temp_requirement,
														it->second.gas_buffer_sizes.tempSizeInBytes);
			batch_nga_ses++;
			if ((batch_build_output_requirement + batch_build_max_temp_requirement + batch_build_compacted_requirement)
				>
					available_mem_pool_size)
				break;
		}

		// d_temp may still be available from a previous batch, but is freed later if it is "too big"
		d_temp.allocIfRequired(batch_build_max_temp_requirement);

		// trash existing buffer if it is more than 10% bigger than what we need
		// if it is roughly the same, we keep it
		if (d_temp_output.byteSize() > batch_build_output_requirement * 1.1)
			d_temp_output.free();
		d_temp_output.allocIfRequired(batch_build_output_requirement);

		// this buffer is assumed to be very small
		// trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
		if (d_temp_compacted_sizes.reservedCount() > batch_nga_ses * 2 && d_temp_compacted_sizes.byteSize() > 20 * 1024
			*
				1024)
			d_temp_compacted_sizes.free();
		d_temp_compacted_sizes.allocIfRequired(batch_nga_ses);

		// sum of build output size of GASes, excluding alignment
		size_t batch_temp_output_size = 0;
		// sum of size of compacted GASes
		size_t batch_compacted_size = 0;

		auto it = gases.rbegin();
		for (size_t i = 0, temp_output_alignment_offset = 0; i < batch_nga_ses; ++i) {
			emitProperty.result = d_temp_compacted_sizes.get(i);
			GasInfo& info = it->second;

			OPTIX_CHECK(optixAccelBuild(context_, nullptr, // CUDA stream
										&accel_options,
										info.build_inputs.data(),
										static_cast<unsigned int>( info.build_inputs.size()),
										d_temp.get(),
										d_temp.byteSize(),
										d_temp_output.get(temp_output_alignment_offset),
										info.gas_buffer_sizes.outputSizeInBytes,
										&info.mesh->gas_handle,
										&emitProperty, // emitted property list
										1 // num emitted properties
			));

			temp_output_alignment_offset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
			it++;
		}

		// trash d_temp if it is at least 20MB in size
		if (d_temp.byteSize() > 20 * 1024 * 1024)
			d_temp.free();

		// download all compacted sizes to allocate final output buffers for these GASes
		std::vector<size_t> h_compacted_sizes(batch_nga_ses);
		d_temp_compacted_sizes.download(h_compacted_sizes.data());

		//////////////////////////////////////////////////////////////////////////
		// TODO:
		// Now we know the actual memory requirement of the compacted GASes.
		// Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
		bool can_compact = false;
		it = gases.rbegin();
		for (size_t i = 0; i < batch_nga_ses; ++i) {
			GasInfo& info = it->second;
			if (info.gas_buffer_sizes.outputSizeInBytes > h_compacted_sizes[i]) {
				can_compact = true;
				break;
			}
			it++;
		}

		if (can_compact) {
			//////////////////////////////////////////////////////////////////////////
			// "batch allocate" the compacted buffers
			it = gases.rbegin();
			for (size_t i = 0; i < batch_nga_ses; ++i) {
				GasInfo& info = it->second;
				batch_compacted_size += h_compacted_sizes[i];
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &info.mesh->d_gas_output ), h_compacted_sizes[i]));
				total_temp_output_processed_size += info.gas_buffer_sizes.outputSizeInBytes;
				it++;
			}

			it = gases.rbegin();
			for (size_t i = 0; i < batch_nga_ses; ++i) {
				GasInfo& info = it->second;
				OPTIX_CHECK(optixAccelCompact(context_, nullptr, info.mesh->gas_handle, info.mesh->d_gas_output,
											  h_compacted_sizes[i], &info.mesh->gas_handle));
				it++;
			}
		} else {
			it = gases.rbegin();
			for (size_t i = 0, temp_output_alignment_offset = 0; i < batch_nga_ses; ++i) {
				GasInfo& info = it->second;
				info.mesh->d_gas_output = d_temp_output.get(temp_output_alignment_offset);
				batch_compacted_size += h_compacted_sizes[i];
				total_temp_output_processed_size += info.gas_buffer_sizes.outputSizeInBytes;

				temp_output_alignment_offset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
				it++;
			}
			d_temp_output.release();
		}

		used_compacted_output_size += batch_compacted_size;

		gases.erase(it.base(), gases.end());
	}
}

void lift::Renderer::buildInstanceAccel(const Scene& scene, int ray_type_count) {
	const auto& meshes = scene.meshes();
	const size_t num_instances = meshes.size();

	std::vector<OptixInstance> optix_instances(num_instances);

	unsigned int sbt_offset = 0;
	for (size_t i = 0; i < meshes.size(); ++i) {
		auto mesh = meshes[i];
		auto& optix_instance = optix_instances[i];
		memset(&optix_instance, 0, sizeof(OptixInstance));

		optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optix_instance.instanceId = static_cast<unsigned int>(i);
		optix_instance.sbtOffset = sbt_offset;
		optix_instance.visibilityMask = 1;
		optix_instance.traversableHandle = mesh->gas_handle;

		memcpy(optix_instance.transform, value_ptr(transpose(mesh->transform)), sizeof(float) * 12);

		sbt_offset += static_cast<unsigned int>(mesh->indices.size()) * ray_type_count;
		// one sbt record per GAS build input per RAY_TYPE
	}

	const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
	CUdeviceptr d_instances;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_instances ), instances_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>( d_instances ),
		optix_instances.data(),
		instances_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	OptixBuildInput instance_input = {};
	instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instance_input.instanceArray.instances = d_instances;
	instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes ias_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context_,
		&accel_options,
		&instance_input,
		1, // num build inputs
		&ias_buffer_sizes
	));

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>( &d_temp_buffer ),
		ias_buffer_sizes.tempSizeInBytes
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>( &d_ias_output_buffer_ ),
		ias_buffer_sizes.outputSizeInBytes
	));

	OPTIX_CHECK(optixAccelBuild(
		context_,
		nullptr, // CUDA stream
		&accel_options,
		&instance_input,
		1, // num build inputs
		d_temp_buffer,
		ias_buffer_sizes.tempSizeInBytes,
		d_ias_output_buffer_,
		ias_buffer_sizes.outputSizeInBytes,
		&ias_handle_,
		nullptr, // emitted property list
		0 // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>( d_temp_buffer )));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>( d_instances )));
}

void lift::Renderer::createPtxModule() {

	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	pipeline_compile_options_ = {};
	pipeline_compile_options_.usesMotionBlur = false;
	pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeline_compile_options_.numPayloadValues = k_NumPayloadValues;
	pipeline_compile_options_.numAttributeValues = 2; // TODO
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";

	const std::string ptx = embedded_ptx_code;

	ptx_module_ = {};
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(
		context_,
		&module_compile_options,
		&pipeline_compile_options_,
		ptx.c_str(),
		ptx.size(),
		log,
		&sizeof_log,
		&ptx_module_
	));
}

void lift::Renderer::createProgramGroups() {
	OptixProgramGroupOptions program_group_options = {};

	char log[2048];
	size_t sizeof_log = sizeof(log);

	//
	// Ray generation
	//
	{
		OptixProgramGroupDesc raygen_prog_group_desc = {};
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = ptx_module_;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__render_frame";

		OPTIX_CHECK(optixProgramGroupCreate(
			context_,
			&raygen_prog_group_desc,
			1, // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&raygen_prog_group_
		)
		);
	}

	//
	// Miss
	//
	{
		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = ptx_module_;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
		sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context_,
			&miss_prog_group_desc,
			1, // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&radiance_miss_group_
		));

		memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = nullptr; // NULL miss program for occlusion rays
		miss_prog_group_desc.miss.entryFunctionName = nullptr;
		sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context_,
			&miss_prog_group_desc,
			1, // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&occlusion_miss_group_
		));
	}

	//
	// Hit group
	//
	OptixProgramGroupDesc hit_prog_group_desc = {};
	hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hit_prog_group_desc.hitgroup.moduleCH = this->ptx_module_;
	hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		this->context_,
		&hit_prog_group_desc,
		1, // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&this->radiance_hit_group_
	));

	memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
	hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hit_prog_group_desc.hitgroup.moduleCH = nullptr;
	hit_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
	sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		this->context_,
		&hit_prog_group_desc,
		1, // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&this->occlusion_hit_group_
	));
}

void lift::Renderer::createPipeline() {
	std::array<OptixProgramGroup, 5> program_groups = {
		raygen_prog_group_,
		radiance_miss_group_,
		occlusion_miss_group_,
		radiance_hit_group_,
		occlusion_hit_group_
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = k_MaxTraceDepth;
	pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	pipeline_link_options.overrideUsesMotionBlur = false;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context_,
		&pipeline_compile_options_,
		&pipeline_link_options,
		program_groups.data(),
		(int)program_groups.size(),
		log,
		&sizeof_log,
		&pipeline_
	));
}

void lift::Renderer::createSbt(const Scene& scene) {
	{
		const size_t raygen_record_size = sizeof(RayGenRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &sbt_.raygenRecord ), raygen_record_size));

		RayGenRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>( sbt_.raygenRecord ),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		));
	}

	{
		const size_t miss_record_size = sizeof(MissDataRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>( &sbt_.missRecordBase ),
			miss_record_size * RAY_TYPE_COUNT
		));

		MissDataRecord ms_sbt[RAY_TYPE_COUNT];
		OPTIX_CHECK(optixSbtRecordPackHeader(radiance_miss_group_, &ms_sbt[0]));
		OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_miss_group_, &ms_sbt[1]));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>( sbt_.missRecordBase ),
			ms_sbt,
			miss_record_size * RAY_TYPE_COUNT,
			cudaMemcpyHostToDevice
		));
		sbt_.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
		sbt_.missRecordCount = RAY_TYPE_COUNT;
	}

	{
		const auto& materials = scene.materials();
		std::vector<HitGroupRecord> hitgroup_records;
		for (const auto& mesh : scene.meshes()) {
			for (size_t i = 0; i < mesh->material_idx.size(); ++i) {
				HitGroupRecord rec = {};
				OPTIX_CHECK(optixSbtRecordPackHeader(radiance_hit_group_, &rec));
				rec.data.geometry_data.type = GeometryData::TRIANGLE_MESH;
				rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
				rec.data.geometry_data.triangle_mesh.normals = mesh->normals[i];
				rec.data.geometry_data.triangle_mesh.tex_coords = mesh->tex_coords[i];
				rec.data.geometry_data.triangle_mesh.indices = mesh->indices[i];

				const int32_t mat_idx = mesh->material_idx[i];
				if (mat_idx >= 0)
					rec.data.material_data = materials[mat_idx];
				else
					rec.data.material_data = MaterialData();
				hitgroup_records.push_back(rec);

				OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_hit_group_, &rec));
				hitgroup_records.push_back(rec);
			}
		}

		const size_t hitgroup_record_size = sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>( &sbt_.hitgroupRecordBase ),
			hitgroup_record_size * hitgroup_records.size()
		));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>( sbt_.hitgroupRecordBase ),
			hitgroup_records.data(),
			hitgroup_record_size * hitgroup_records.size(),
			cudaMemcpyHostToDevice
		));

		sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
		sbt_.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
	}

}
