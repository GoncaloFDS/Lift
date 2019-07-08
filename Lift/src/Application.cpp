#include "pch.h"
#include "Application.h"

#include "Log.h"

#include "ImGui/ImguiLayer.h"

//Temporary
#include <glad/glad.h>
#include "Renderer/Texture.h"
#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/Optix/OptixErrorCodes.h"
#include "Renderer/Renderer.h"
#include "Renderer/RenderCommand.h"
#include "Cuda/ParallelogramLight.cuh"
#include <fstream>

namespace lift {
	////////////////////////////////////////////////////

	optix::float3 camera_eye = optix::make_float3(278, 273, -900);
	optix::float3 camera_look_at = optix::make_float3(278, 273, 0);
	optix::float3 camera_up = optix::make_float3(0, 1, 0);
	optix::Matrix4x4 camera_rotate = optix::Matrix4x4::identity();

	optix::Program program_intersection = nullptr;
	optix::Program program_bounding_box = nullptr;

	std::string GetPtxString(const char* file_name) {
		std::string ptx_source;

		const std::ifstream file(file_name);
		if (file.good()) {
			std::stringstream source_buffer;
			source_buffer << file.rdbuf();
			return source_buffer.str();
		}
		LF_CORE_ERROR("Invalid PTX path: {0}", file_name);
		return "Invalid PTX";
	}

	optix::Buffer CreateOptixBuffer(optix::Context context, RTformat format, unsigned width, unsigned height,
									bool use_pbo, RTbuffertype buffer_type) {
		optix::Buffer buffer;
		if (use_pbo) {
			const unsigned int element_size = format == RT_FORMAT_UNSIGNED_BYTE4 ? 4 : 16;

			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, element_size * width * height, nullptr, GL_STREAM_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			buffer = context->createBufferFromGLBO(buffer_type, vbo);
			buffer->setFormat(format);
			buffer->setSize(width, height);
		}
		else {
			buffer = context->createBuffer(buffer_type, format, width, height);
		}
		return buffer;
	}

	void setMaterial(optix::GeometryInstance& gi, optix::Material material, const std::string& color_name,
					 const optix::float3& color) {
		gi->addMaterial(material);
		gi[color_name]->setFloat(color);
	}

	optix::GeometryInstance Application::CreateParallelogram(const optix::float3& anchor, const optix::float3& offset1,
															 const optix::float3& offset2) {
		optix::Geometry parallelogram = optix_context_->createGeometry();
		parallelogram->setPrimitiveCount(1u);
		parallelogram->setIntersectionProgram(program_intersection);
		parallelogram->setBoundingBoxProgram(program_bounding_box);

		optix::float3 normal = optix::normalize(optix::cross(offset1, offset2));
		float d = dot(normal, anchor);
		optix::float4 plane = make_float4(normal, d);

		optix::float3 v1 = offset1 / optix::dot(offset1, offset1);
		optix::float3 v2 = offset2 / optix::dot(offset2, offset2);

		parallelogram["plane"]->setFloat(plane);
		parallelogram["anchor"]->setFloat(anchor);
		parallelogram["v1"]->setFloat(v1);
		parallelogram["v2"]->setFloat(v2);

		optix::GeometryInstance gi = optix_context_->createGeometryInstance();
		gi->setGeometry(parallelogram);
		return gi;
	}

	void CalculateCameraVariables(optix::float3 eye, optix::float3 look_at, optix::float3 up, float fov,
								  float aspect_ratio, optix::float3& U, optix::float3& V, optix::float3& W,
								  bool fov_is_vertical) {
		W = look_at - eye; // Do not normalize W -- it implies focal length

		const float w_len = length(W);
		U = normalize(cross(W, up));
		V = normalize(cross(U, W));
		if (fov_is_vertical) {
			const float v_len = w_len * tanf(0.5f * fov * M_PIf / 180.0f);
			V *= v_len;
			const float u_len = v_len * aspect_ratio;
			U *= u_len;
		}
	}

	void Application::UpdateCamera() {
		const float fov = 35.0f;
		const float aspect_ratio = window_->GetWidth() / window_->GetHeight();

		optix::float3 camera_u, camera_v, camera_w;
		CalculateCameraVariables(camera_eye, camera_look_at, camera_up, fov, aspect_ratio,
								 camera_u, camera_v, camera_w, true);

		const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
			normalize(camera_u),
			normalize(camera_v),
			normalize(-camera_w),
			camera_look_at);

		const optix::Matrix4x4 frame_inv = frame.inverse();

		// Apply camera rotation twice to match old SDK behavior
		const optix::Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

		camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
		camera_look_at = optix::make_float3(trans * optix::make_float4(camera_look_at, 1.0f));
		camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

		CalculateCameraVariables(
			camera_eye, camera_look_at, camera_up, fov, aspect_ratio,
			camera_u, camera_v, camera_w, true);

		camera_rotate = optix::Matrix4x4::identity();

		/*if (camera_changed) // reset accumulation
			frame_number = 1;
		camera_changed = false;*/

		optix_context_["frame_number"]->setUint(1);
		optix_context_["eye"]->setFloat(camera_eye);
		optix_context_["U"]->setFloat(camera_u);
		optix_context_["V"]->setFloat(camera_v);
		optix_context_["W"]->setFloat(camera_w);
	}

	//////////////////////////////////////////////////
	Application* Application::instance_ = nullptr;

	Application::Application() {
		LF_CORE_ASSERT(!instance_, "Application already exists");
		instance_ = this;
		window_ = std::unique_ptr<Window>(Window::Create());
		window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));
		//window_->SetVSync(false);

		InitGraphicsContext();
		InitOptix();

		PushOverlay<ImGuiLayer>();
	}

	void Application::Run() {
		CreateScene();
		optix_context_->validate();

		while (is_running_) {
			ImGuiLayer::Begin();
			UpdateCamera();

			RenderCommand::SetClearColor({0.1f, 0.1f, 0.1f, 0.0f});
			RenderCommand::Clear();
			optix_context_->launch(0, window_->GetWidth(), window_->GetHeight());

			Renderer::BeginScene();

			//const Texture texture(optix_context_["output_buffer"]->getBuffer());
			//texture.Bind();
			shader_->Bind();
			//shader_->SetUniform1i("u_Texture", 0);
			Renderer::Submit(vertex_array_);
			Renderer::EndScene();


			for (auto& layer : layer_stack_)
				layer->OnUpdate();

			for (auto& layer : layer_stack_)
				layer->OnImguiRender();
			EndFrame();
		}
	}

	void Application::CreateScene() {

		float vertices [4 * 5] = {
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
		};

		vertex_array_.reset(VertexArray::Create());
		std::shared_ptr<VertexBuffer> vertex_buffer{};
		vertex_buffer.reset(VertexBuffer::Create(vertices, sizeof(vertices)));

		vertex_buffer->SetLayout({
			{ShaderDataType::Float3, "a_Position"},
			{ShaderDataType::Float2, "a_Uv"}
		});

		vertex_array_->AddVertexBuffer(vertex_buffer);

		uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
		std::shared_ptr<IndexBuffer> index_buffer;
		index_buffer.reset(IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t)));
		vertex_array_->SetIndexBuffer(index_buffer);

		const Texture texture(std::string("res/textures/test.png"));
		texture.Bind();
		shader_ = std::make_unique<Shader>("res/shaders/default");
		shader_->Bind();
		shader_->SetUniform1i("u_Texture", 0);

		//////////////////////////////////
		///
		///
		//////////////////////////////////

		ParallelogramLight light{};
		light.corner = optix::make_float3(343, 548, 227);
		light.v1 = optix::make_float3(-130, 0.0f, 0.0f);
		light.v2 = optix::make_float3(0.0f, 0.0f, 105.0f);
		light.normal = optix::normalize(optix::cross(light.v1, light.v2));
		light.emission = optix::make_float3(15.0f, 15.0f, 5.0f);

		optix::Buffer light_buffer = optix_context_->createBuffer(RT_BUFFER_INPUT);
		light_buffer->setFormat(RT_FORMAT_USER);
		light_buffer->setElementSize(sizeof(ParallelogramLight));
		light_buffer->setSize(1u);
		memcpy(light_buffer->map(), &light, sizeof(light));
		light_buffer->unmap();
		optix_context_["lights"]->setBuffer(light_buffer);

		// Set up Material
		optix::Material diffuse = optix_context_->createMaterial();
		std::string ptx_string = GetPtxString("res/ptx/optixPathTracer.ptx");
		const optix::Program diffuse_ch = optix_context_->createProgramFromPTXString(ptx_string, "diffuse");
		const optix::Program diffuse_ah = optix_context_->createProgramFromPTXString(ptx_string, "shadow");
		diffuse->setClosestHitProgram(0, diffuse_ch);
		diffuse->setAnyHitProgram(1, diffuse_ah);

		optix::Material diffuse_light = optix_context_->createMaterial();
		optix::Program diffuse_em = optix_context_->createProgramFromPTXString(ptx_string, "diffuseEmitter");
		diffuse_light->setClosestHitProgram(0, diffuse_em);

		// Set up parallelogram programs
		ptx_string = GetPtxString("res/ptx/parallelogram.ptx");
		program_bounding_box = optix_context_->createProgramFromPTXString(ptx_string, "bounds");
		program_intersection = optix_context_->createProgramFromPTXString(ptx_string, "intersect");

		// Create geometry Instances
		std::vector<optix::GeometryInstance> geometry_instances;
		const optix::float3 white = optix::make_float3(0.8f, 0.8f, 0.8f);
		const optix::float3 green = optix::make_float3(0.05f, 0.8f, 0.05f);
		const optix::float3 red = optix::make_float3(0.8f, 0.05f, 0.05f);
		const optix::float3 light_em = optix::make_float3(15.0f, 15.0f, 5.0f);

		// Floor
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(0.0f, 0.0f, 0.0f),
														 optix::make_float3(0.0f, 0.0f, 559.2f),
														 optix::make_float3(556.0f, 0.0f, 0.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);

		// Ceiling
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(0.0f, 548.8f, 0.0f),
														 optix::make_float3(556.0f, 0.0f, 0.0f),
														 optix::make_float3(0.0f, 0.0f, 559.2f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);

		// Back wall
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(0.0f, 0.0f, 559.2f),
														 optix::make_float3(0.0f, 548.8f, 0.0f),
														 optix::make_float3(556.0f, 0.0f, 0.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);

		// Right wall
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(0.0f, 0.0f, 0.0f),
														 optix::make_float3(0.0f, 548.8f, 0.0f),
														 optix::make_float3(0.0f, 0.0f, 559.2f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", green);

		// Left wall
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(556.0f, 0.0f, 0.0f),
														 optix::make_float3(0.0f, 0.0f, 559.2f),
														 optix::make_float3(0.0f, 548.8f, 0.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", red);

		// Short block
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(130.0f, 165.0f, 65.0f),
														 optix::make_float3(-48.0f, 0.0f, 160.0f),
														 optix::make_float3(160.0f, 0.0f, 49.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(290.0f, 0.0f, 114.0f),
														 optix::make_float3(0.0f, 165.0f, 0.0f),
														 optix::make_float3(-50.0f, 0.0f, 158.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(130.0f, 0.0f, 65.0f),
														 optix::make_float3(0.0f, 165.0f, 0.0f),
														 optix::make_float3(160.0f, 0.0f, 49.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(82.0f, 0.0f, 225.0f),
														 optix::make_float3(0.0f, 165.0f, 0.0f),
														 optix::make_float3(48.0f, 0.0f, -160.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(240.0f, 0.0f, 272.0f),
														 optix::make_float3(0.0f, 165.0f, 0.0f),
														 optix::make_float3(-158.0f, 0.0f, -47.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);

		// Tall block
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(423.0f, 330.0f, 247.0f),
														 optix::make_float3(-158.0f, 0.0f, 49.0f),
														 optix::make_float3(49.0f, 0.0f, 159.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(423.0f, 0.0f, 247.0f),
														 optix::make_float3(0.0f, 330.0f, 0.0f),
														 optix::make_float3(49.0f, 0.0f, 159.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(472.0f, 0.0f, 406.0f),
														 optix::make_float3(0.0f, 330.0f, 0.0f),
														 optix::make_float3(-158.0f, 0.0f, 50.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(314.0f, 0.0f, 456.0f),
														 optix::make_float3(0.0f, 330.0f, 0.0f),
														 optix::make_float3(-49.0f, 0.0f, -160.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(265.0f, 0.0f, 296.0f),
														 optix::make_float3(0.0f, 330.0f, 0.0f),
														 optix::make_float3(158.0f, 0.0f, -49.0f)));
		setMaterial(geometry_instances.back(), diffuse, "diffuse_color", white);

		// Create shadow group (no light)
		optix::GeometryGroup shadow_group = optix_context_->createGeometryGroup(
			geometry_instances.begin(), geometry_instances.end());
		shadow_group->setAcceleration(optix_context_->createAcceleration("Trbvh"));
		optix_context_["top_shadower"]->set(shadow_group);

		// Light
		geometry_instances.push_back(CreateParallelogram(optix::make_float3(343.0f, 548.6f, 227.0f),
														 optix::make_float3(-130.0f, 0.0f, 0.0f),
														 optix::make_float3(0.0f, 0.0f, 105.0f)));
		setMaterial(geometry_instances.back(), diffuse_light, "emission_color", light_em);

		// Create geometry group
		optix::GeometryGroup geometry_group = optix_context_->createGeometryGroup(
			geometry_instances.begin(), geometry_instances.end());
		geometry_group->setAcceleration(optix_context_->createAcceleration("Trbvh"));
		optix_context_["top_object"]->set(geometry_group);

	}

	void Application::InitOptix() {
		GetOptixSystemInformation();

		optix_context_ = optix::Context::create();
		optix_context_->setRayTypeCount(2);
		optix_context_->setEntryPointCount(1);
		optix_context_->setStackSize(1800);
		optix_context_->setMaxTraceDepth(2);

		optix_context_["scene_epsilon"]->setFloat(1.e-3f);
		optix_context_["rr_begin_depth"]->setUint(1);

		optix::Buffer buffer = CreateOptixBuffer(optix_context_, RT_FORMAT_FLOAT4, window_->GetWidth(),
												 window_->GetHeight(), true, RT_BUFFER_OUTPUT);
		optix_context_["output_buffer"]->set(buffer);

		const std::string ptx_string = GetPtxString("res/ptx/optixPathTracer.ptx");
		optix_context_->setRayGenerationProgram(
			0, optix_context_->createProgramFromPTXString(ptx_string, "pathtrace_camera"));
		optix_context_->setExceptionProgram(0, optix_context_->createProgramFromPTXString(ptx_string, "exception"));
		optix_context_->setMissProgram(0, optix_context_->createProgramFromPTXString(ptx_string, "miss"));

		//optix_context_->validate();
		optix_context_["sqrt_num_samples"]->setUint(2);
		optix_context_["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f);
		optix_context_["bg_color"]->setFloat(0.0f, 0.0f, 0.f);

	}

	void Application::InitGraphicsContext() {
		graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
		graphics_context_->Init();
	}

	void Application::InitPrograms() {
		//ptx_programs_["raygeneration"] = optix_context_->createProgramFromPTXFile("raygeneration.ptx", "raygeneration");

	}

	void Application::EndFrame() const {
		ImGuiLayer::End();
		graphics_context_->SwapBuffers();
		window_->OnUpdate();
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));

		for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
			(*--it)->OnEvent(e);
			if (e.handled_)
				break;
		}
	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		is_running_ = false;

		return true;
	}

	void Application::GetOptixSystemInformation() {
		unsigned int optix_version;
		OPTIX_CALL(rtGetVersion(&optix_version));

		const unsigned int major = optix_version / 10000;
		const unsigned int minor = (optix_version % 10000) / 100;
		const unsigned int micro = optix_version % 100;
		LF_CORE_INFO("");
		LF_CORE_INFO("Optix Info:");
		LF_CORE_INFO("\tVersion: {0}.{1}.{2}", major, minor, micro);

		unsigned int number_of_devices = 0;
		OPTIX_CALL(rtDeviceGetDeviceCount(&number_of_devices));
		LF_CORE_INFO("\tNumber of Devices = {0}", number_of_devices);

		for (unsigned int i = 0; i < number_of_devices; ++i) {
			char name[256];
			OPTIX_CALL(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
			LF_CORE_INFO("\tDevice {0}: {1}", i, name);

			int compute_capability[2] = {0, 0};
			OPTIX_CALL(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(
				compute_capability), &compute_capability));
			LF_CORE_INFO("\t\tCompute Support: {0}.{1}", compute_capability[0], compute_capability[1]);
		}
	}
}
