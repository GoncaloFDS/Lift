#include "pch.h"
#include "ImGuiLayer.h"

#include "imgui.h"

#include "Application.h"
#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_opengl3.h"

// TEMPORARY
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include "imgui_internal.h"

lift::ImGuiLayer::ImGuiLayer()
	: Layer("ImGuiLayer") {

}

lift::ImGuiLayer::~ImGuiLayer() {
	ImGuiLayer::OnDetach();
};

void lift::ImGuiLayer::OnAttach() {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	auto& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows

	ImGui::StyleColorsDark();

	// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	auto& app = Application::Get();
	auto* window = static_cast<GLFWwindow*>(app.GetWindow().GetNativeWindow());

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 410");


}

void lift::ImGuiLayer::OnDetach() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void lift::ImGuiLayer::OnUpdate() {
}

void lift::ImGuiLayer::OnImguiRender() {
	static float f = 0.0f;
	static int counter = 0;

	auto& app = Application::Get();
	ImGui::Begin("Begin Window");
	ImGui::Text("This is some useful text.");
	ImGui::Checkbox("Demo Window", &show_demo_window_);
	ImGui::Checkbox("Another Window", &show_another_window_);

	ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
	ImGui::ColorEdit3("Top color", &app.GetTopColor().x);
	ImGui::ColorEdit3("Bottom color", &app.GetBottomColor().x);
	if (ImGui::Button("Button"))
		counter++;
	ImGui::SameLine();
	ImGui::Text("counter = %d", counter);

	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
				ImGui::GetIO().Framerate);
	ImGui::End();

}

void lift::ImGuiLayer::OnEvent(Event& event) {
	auto& io = ImGui::GetIO();
	if(io.WantCaptureMouse)
		event.handled_ = true;

}

void lift::ImGuiLayer::Begin() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

}

void lift::ImGuiLayer::End() {
	auto& io = ImGui::GetIO();
	auto& app = Application::Get();
	io.DisplaySize = ImVec2(static_cast<float>(app.GetWindow().GetWidth()),
							static_cast<float>(app.GetWindow().GetHeight()));

	// Rendering
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		auto backup_current_context = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backup_current_context);
	}
}
