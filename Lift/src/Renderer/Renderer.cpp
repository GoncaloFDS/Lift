#include "pch.h"
#include "Renderer.h"
#include "RenderCommand.h"
#include <optix_stubs.h>

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
		int objectID;
	};
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
	if (launch_params_.frame_buffer_size.x == 0) {
		LF_CORE_ERROR("frame buffer size is zero");
		return;
	}
	launch_params_buffer_.upload(&launch_params_, 1);
	launch_params_.frame_id++;

	OPTIX_CHECK(optixLaunch(pipeline_,
		stream_,
		launch_params_buffer_.d_pointer(),
		launch_params_buffer_.size_in_bytes,
		&sbt_,
		launch_params_.frame_buffer_size.x,
		launch_params_.frame_buffer_size.y,
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

	launch_params_.frame_buffer_size = size;
	launch_params_.color_buffer = static_cast<uint32_t*>(color_buffer_.d_ptr);
}

void lift::Renderer::DownloadPixels(uint32_t h_pixels[]) {
	color_buffer_.download(h_pixels, launch_params_.frame_buffer_size.x * launch_params_.frame_buffer_size.y);
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
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygen_program_groups_.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_groups_[i],&rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}
	raygen_program_groups_buffer_.alloc_and_upload(raygenRecords);
	sbt_.raygenRecord = raygen_program_groups_buffer_.d_pointer();

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < miss_program_groups_.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_groups_[i],&rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	miss_program_groups_buffer_.alloc_and_upload(missRecords);
	sbt_.missRecordBase = miss_program_groups_buffer_.d_pointer();
	sbt_.missRecordStrideInBytes = sizeof(MissRecord);
	sbt_.missRecordCount = (int)missRecords.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

	// we don't actually have any objects in this example, but let's
	// create a dummy one so the SBT doesn't have any null pointers
	// (which the sanity checks in compilation would compain about)
	int numObjects = 1;
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < numObjects; i++) {
		int object_type = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hit_program_groups_[object_type],&rec));
		rec.objectID = i;
		hitgroupRecords.push_back(rec);
	}
	hit_program_groups_buffer_.alloc_and_upload(hitgroupRecords);
	sbt_.hitgroupRecordBase = hit_program_groups_buffer_.d_pointer();
	sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt_.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void lift::Renderer::Submit(const std::shared_ptr<VertexArray>& vertex_array) {
	vertex_array->Bind();
	RenderCommand::DrawIndexed(vertex_array);
}
