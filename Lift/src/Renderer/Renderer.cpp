#include "pch.h"
#include "Renderer.h"
#include "RenderCommand.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


namespace lift {
	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		int object_id;
	};
}

void lift::TriangleMesh::AddCube(const vec3& center, const vec3& size) {
	const int first_vertex_index = int(vertices.size());
	vertices.emplace_back(0.0f, 0.0f, 0.0f);
	vertices.emplace_back(1.0f, 0.0f, 0.0f);
	vertices.emplace_back(0.0f, 1.0f, 0.0f);
	vertices.emplace_back(1.0f, 1.0f, 0.0f);
	vertices.emplace_back(0.0f, 0.0f, 1.0f);
	vertices.emplace_back(1.0f, 0.0f, 1.0f);
	vertices.emplace_back(0.0f, 1.0f, 1.0f);
	vertices.emplace_back(1.0f, 1.0f, 1.0f);

	int indices_data[] = {
		0, 1, 3, 2, 3, 0,
		5, 7, 6, 5, 6, 4,
		0, 4, 5, 0, 5, 1,
		2, 3, 7, 2, 7, 6,
		1, 5, 6, 1, 7, 3,
		4, 0, 2, 4, 2, 6
	};
	for (int i = 0; i < 12; i++) {
		indices.push_back(ivec3(first_vertex_index) + ivec3(indices_data[3 * i + 0],
															indices_data[3 * i + 1],
															indices_data[3 * i + 2]));
	}
}

void lift::Renderer::Init() {

	InitOptix();
	CreateContext();
	CreateModule();

	CreateRaygenPrograms();
	CreateMissPrograms();
	CreateHitgroupPrograms();

	CreatePipeline();
	BuildShaderBindingTables();

	launch_params_buffer_.alloc(sizeof(launch_params_));
}

void lift::Renderer::Render() {
	if (launch_params_.frame.size.x == 0) {
		LF_CORE_ERROR("frame buffer size is zero");
		return;
	}
	launch_params_buffer_.upload(&launch_params_, 1);

	OPTIX_CHECK(optixLaunch(pipeline_,
		stream_,
		launch_params_buffer_.d_pointer(),
		launch_params_buffer_.size_in_bytes,
		&sbt_,
		launch_params_.frame.size.x,
		launch_params_.frame.size.y,
		1
	));
	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	CUDA_SYNC_CHECK();
}

void lift::Renderer::Resize(const ivec2& size) {
	color_buffer_.resize(size.x * size.y * sizeof(uint32_t));

	launch_params_.frame.size = size;
	launch_params_.frame.color_buffer = static_cast<uint32_t*>(color_buffer_.d_ptr);

	SetCamera(last_set_camera_);
}

void lift::Renderer::DownloadPixels(uint32_t h_pixels[]) {
	color_buffer_.download(h_pixels, launch_params_.frame.size.x * launch_params_.frame.size.y);
}

void lift::Renderer::SetCamera(const Camera& camera) {
	last_set_camera_ = camera;
	launch_params_.camera.position = camera.from;
	launch_params_.camera.direction = normalize(camera.at - camera.from);

	const float cos_fov_y = 0.66f;
	const float aspect = launch_params_.frame.size.x / float(launch_params_.frame.size.y);
	launch_params_.camera.horizontal = cos_fov_y * aspect *
		normalize(cross(launch_params_.camera.direction, camera.up));
	launch_params_.camera.vertical = cos_fov_y * normalize(cross(launch_params_.camera.horizontal,
																 launch_params_.camera.direction));
}

void lift::Renderer::AddModel(const TriangleMesh& model) {
	launch_params_.traversable = BuildAccelerationStructure(model);
}

void lift::Renderer::InitOptix() {
	LF_CORE_INFO("Initializing optix...");
	cudaFree(0);
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	LF_ASSERT(num_devices > 0, "No CUDA capable device found");
	LF_CORE_INFO("Found {0} CUDA devices", num_devices);
	OPTIX_CHECK(optixInit());

	LF_CORE_INFO("Successfully initialized optix...");
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
	//fprintf(stderr, "[%2d][%12s]: %s\n", level, tag, message);
}

void lift::Renderer::CreateContext() {
	const auto device_id = 0;
	CUDA_CHECK(SetDevice(device_id));
	CUDA_CHECK(StreamCreate(&stream_));

	cudaGetDeviceProperties(&device_props_, device_id);

	const CUresult cuda_result = cuCtxGetCurrent(&cuda_context_);
	if (cuda_result != CUDA_SUCCESS)
		LF_CORE_ERROR("Error querying current context: error code {0}", cuda_result);

	OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, nullptr, &optix_context_));
	OPTIX_CHECK(optixDeviceContextSetLogCallback (optix_context_, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
void lift::Renderer::CreateModule() {
	module_compile_options_.maxRegisterCount = 100;
	module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	pipeline_compile_options_ = {};
	pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipeline_compile_options_.usesMotionBlur = false;
	pipeline_compile_options_.numPayloadValues = 2;
	pipeline_compile_options_.numAttributeValues = 2;
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeline_compile_options_.pipelineLaunchParamsVariableName = "optix_launch_parameters";

	pipeline_link_options_.overrideUsesMotionBlur = false;
	pipeline_link_options_.maxTraceDepth = 2;

	const auto ptx_code = Util::GetPtxString("res/ptx/device_programs.ptx");

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
		&module_compile_options_,
		&pipeline_compile_options_,
		ptx_code.c_str(),
		ptx_code.size(),
		log,
		&sizeof_log,
		&module_
	));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);
}

void lift::Renderer::CreateRaygenPrograms() {
	raygen_program_groups_.resize(1);

	OptixProgramGroupOptions program_group_options = {};
	OptixProgramGroupDesc program_group_desc = {};
	program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	program_group_desc.raygen.module = module_;
	program_group_desc.raygen.entryFunctionName = "__raygen__render_frame";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
		&program_group_desc,
		1,
		&program_group_options,
		log,&sizeof_log,
		&raygen_program_groups_[0]
	));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);
}

void lift::Renderer::CreateMissPrograms() {
	miss_program_groups_.resize(1);

	OptixProgramGroupOptions program_group_options = {};
	OptixProgramGroupDesc program_group_desc = {};
	program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	program_group_desc.raygen.module = module_;
	program_group_desc.raygen.entryFunctionName = "__miss__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
		&program_group_desc,
		1,
		&program_group_options,
		log,&sizeof_log,
		&miss_program_groups_[0]
	));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);

}

void lift::Renderer::CreateHitgroupPrograms() {
	// for this simple example, we set up a single hit group
	hit_program_groups_.resize(1);

	OptixProgramGroupOptions program_group_options = {};
	OptixProgramGroupDesc program_group_desc = {};
	program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	program_group_desc.hitgroup.moduleCH = module_;
	program_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	program_group_desc.hitgroup.moduleAH = module_;
	program_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
		&program_group_desc,
		1,
		&program_group_options,
		log,&sizeof_log,
		&hit_program_groups_[0]
	));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);
}

void lift::Renderer::CreatePipeline() {
	std::vector<OptixProgramGroup> program_groups;
	for (auto program_group : raygen_program_groups_)
		program_groups.push_back(program_group);
	for (auto program_group : miss_program_groups_)
		program_groups.push_back(program_group);
	for (auto program_group : hit_program_groups_)
		program_groups.push_back(program_group);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optix_context_,
		&pipeline_compile_options_,
		&pipeline_link_options_,
		program_groups.data(),
		static_cast<int>(program_groups.size()),
		log,&sizeof_log,
		&pipeline_
	));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);

	OPTIX_CHECK(optixPipelineSetStackSize
		( /* [in] The pipeline to configure the stack size for */
			pipeline_,
			/* [in] The direct stack size requirement for direct
			   callables invoked from IS or AH. */
			2*1024,
			/* [in] The direct stack size requirement for direct
			   callables invoked from RG, MS, or CH.  */
			2*1024,
			/* [in] The continuation stack requirement. */
			2*1024,
			/* [in] The maximum depth of a traversable graph
			   passed to trace. */
			3));
	if (sizeof_log > 1)
		LF_CORE_WARN(log);
}

void lift::Renderer::BuildShaderBindingTables() {
	// ------------------------------------------------------------------
	// build raygen records
	// ------------------------------------------------------------------
	std::vector<RaygenRecord> raygen_records;
	for (int i = 0; i < raygen_program_groups_.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_groups_[i],&rec));
		rec.data = nullptr; /* for now ... */
		raygen_records.push_back(rec);
	}
	raygen_program_groups_buffer_.alloc_and_upload(raygen_records);
	sbt_.raygenRecord = raygen_program_groups_buffer_.d_pointer();

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	std::vector<MissRecord> miss_records;
	for (int i = 0; i < miss_program_groups_.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_groups_[i],&rec));
		rec.data = nullptr; /* for now ... */
		miss_records.push_back(rec);
	}
	miss_program_groups_buffer_.alloc_and_upload(miss_records);
	sbt_.missRecordBase = miss_program_groups_buffer_.d_pointer();
	sbt_.missRecordStrideInBytes = sizeof(MissRecord);
	sbt_.missRecordCount = (int)miss_records.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

	// we don't actually have any objects in this example, but let's
	// create a dummy one so the SBT doesn't have any null pointers
	// (which the sanity checks in compilation would compain about)
	const int num_objects = 1;
	std::vector<HitgroupRecord> hitgroup_records;
	for (int i = 0; i < num_objects; i++) {
		const int object_type = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hit_program_groups_[object_type],&rec));
		rec.object_id = i;
		hitgroup_records.push_back(rec);
	}
	hit_program_groups_buffer_.alloc_and_upload(hitgroup_records);
	sbt_.hitgroupRecordBase = hit_program_groups_buffer_.d_pointer();
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt_.hitgroupRecordCount = (int)hitgroup_records.size();
}

OptixTraversableHandle lift::Renderer::BuildAccelerationStructure(const TriangleMesh& model) {
	vertices_buffer_.alloc_and_upload(model.vertices);
	indices_buffer_.alloc_and_upload(model.indices);

	OptixTraversableHandle acceleration_struct{0};

	// ==================================================================
	// triangle inputs
	// ==================================================================
	OptixBuildInput triangle_input{};
	triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	auto d_vertices = vertices_buffer_.d_pointer();
	auto d_indices = indices_buffer_.d_pointer();

	triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.vertexStrideInBytes = sizeof(vec3);
	triangle_input.triangleArray.numVertices = int(model.vertices.size());
	triangle_input.triangleArray.vertexBuffers = &d_vertices;

	triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangle_input.triangleArray.indexStrideInBytes = sizeof(ivec3);
	triangle_input.triangleArray.numIndexTriplets = int(model.indices.size());
	triangle_input.triangleArray.indexBuffer = d_indices;

	uint32_t triangle_input_flags[1] = {0};

	triangle_input.triangleArray.flags = triangle_input_flags;
	triangle_input.triangleArray.numSbtRecords = 1;
	triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
	triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	// ==================================================================
	// BLAS setup
	// ==================================================================
	OptixAccelBuildOptions acceleration_options = {};
	acceleration_options.buildFlags = OPTIX_BUILD_FLAG_NONE
		| OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	acceleration_options.motionOptions.numKeys = 1;
	acceleration_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blas_buffer_sizes;
	OPTIX_CHECK(
		optixAccelComputeMemoryUsage (optix_context_, &acceleration_options, &triangle_input, 1, &blas_buffer_sizes));

	// ==================================================================
	// prepare compaction
	// ==================================================================	
	CudaBuffer compacted_size_buffer;
	compacted_size_buffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = compacted_size_buffer.d_pointer();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CudaBuffer temp_buffer;
	temp_buffer.alloc(blas_buffer_sizes.tempSizeInBytes);

	CudaBuffer output_buffer;
	output_buffer.alloc(blas_buffer_sizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optix_context_,
		nullptr,
		&acceleration_options,
		&triangle_input,
		1,
		temp_buffer.d_pointer(),
		temp_buffer.size_in_bytes,
		output_buffer.d_pointer(),
		output_buffer.size_in_bytes,
		&acceleration_struct,
		&emit_desc,1
	));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compacted_size;
	compacted_size_buffer.download(&compacted_size, 1);

	acceleration_struct_buffer_.alloc(compacted_size);
	OPTIX_CHECK(optixAccelCompact(optix_context_,
		/*stream:*/nullptr,
		acceleration_struct,
		acceleration_struct_buffer_.d_pointer(),
		acceleration_struct_buffer_.size_in_bytes,
		&acceleration_struct));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	output_buffer.free(); // << the UNcompacted, temporary output buffer
	temp_buffer.free();
	compacted_size_buffer.free();

	return acceleration_struct;

}

void lift::Renderer::Submit(const std::shared_ptr<VertexArray>& vertex_array) {
	vertex_array->Bind();
	RenderCommand::DrawIndexed(vertex_array);
}
