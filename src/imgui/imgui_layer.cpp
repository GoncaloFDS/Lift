#include "pch.h"
#include "imgui_layer.h"

#include "imgui.h"

#include "application.h"
#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"
#include "imgui_internal.h"
#include <core/profiler.h>

ivec2 lift::ImGuiLayer::s_render_window_size;

// TEMPORARY
ImGuiWindowFlags k_WindowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove;

lift::ImGuiLayer::ImGuiLayer()
    : Layer("ImGuiLayer") {

}

lift::ImGuiLayer::~ImGuiLayer() {
    ImGuiLayer::onDetach();
};

void lift::ImGuiLayer::onAttach() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    //io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    //io.IniFilename = "res/imgui.ini";

    // Setup platform/Renderer bindings
    auto& app = Application::get();
    auto* window_handle = static_cast<GLFWwindow*>(app.getWindow().getNativeWindow());
    ImGui_ImplGlfw_InitForOpenGL(window_handle, true);
    ImGui_ImplOpenGL3_Init("#version 410");

}

void lift::ImGuiLayer::onDetach() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void lift::ImGuiLayer::onUpdate() {
}

void lift::ImGuiLayer::onImguiRender() {

    auto& app = Application::get();
    if (ImGui::Begin("Editor")) {

        ImGui::Text("Frame Size: %dx%d ", app.getWindow().width(), app.getWindow().height());

        ImGui::Separator();
        ImGui::Text("Frame Duration %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Profiler")) {
            ImGui::Text("Scene Update          %.6f ms", Profiler::getDuration(Profiler::Id::SceneUpdate) * 1000.0f);
            ImGui::Text("Scene Render          %.6f ms", Profiler::getDuration(Profiler::Id::Render) * 1000.0f);
            ImGui::Text("Display               %.6f ms", Profiler::getDuration(Profiler::Id::Display) * 1000.0f);
        }
    }
    ImGui::End();
}

void lift::ImGuiLayer::onEvent(Event& event) {
    const auto& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        event.handled = true;
    }
}

void lift::ImGuiLayer::begin() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void lift::ImGuiLayer::end() {
    auto& io = ImGui::GetIO();
    auto& app = Application::get();
    io.DisplaySize = ImVec2(static_cast<float>(app.getWindow().width()),
                            static_cast<float>(app.getWindow().height()));
    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}
