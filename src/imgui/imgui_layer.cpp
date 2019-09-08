#include "pch.h"
#include "imgui_layer.h"

#include "imgui.h"

#include "application.h"
#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"
#include "imgui_internal.h"

ivec2 lift::ImGuiLayer::render_window_size;

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
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows

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
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_NoResize;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
        window_flags |= ImGuiWindowFlags_NoBackground;

    static bool p_open = true;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", &p_open, window_flags);
    ImGui::PopStyleVar();

    ImGui::PopStyleVar(2);

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    } else {
    }

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Docking")) {
            if (ImGui::MenuItem("Flag: NoSplit", "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0))
                dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;
            if (ImGui::MenuItem("Flag: NoResize", "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0))
                dockspace_flags ^= ImGuiDockNodeFlags_NoResize;
            if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "",
                                (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0))
                dockspace_flags ^=
                    ImGuiDockNodeFlags_NoDockingInCentralNode;
            if (ImGui::MenuItem("Flag: PassthruCentralNode", "",
                                (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0))
                dockspace_flags ^=
                    ImGuiDockNodeFlags_PassthruCentralNode;
            if (ImGui::MenuItem("Flag: AutoHideTabBar", "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0))
                dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;
            ImGui::Separator();
            if (ImGui::MenuItem("Close DockSpace", nullptr, false, p_open != false))
                p_open = false;
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    ImGui::End();

    auto& app = Application::get();
    ImGui::Begin("Editor");
    //ImGui::ColorEdit3("Top color", &app.GetTopColor().x);
    //ImGui::ColorEdit3("Bottom color", &app.GetBottomColor().x);
    if (ImGui::ColorEdit3("Albedo", &app.material_albedo.x))
        app.restartAccumulation();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Begin("Render");
    const auto window = ImGui::GetCurrentWindow();
    auto size = ivec2(window->Size.x, window->Size.y);
    if (size != render_window_size) {
        app.resize(size);
        render_window_size = size;
    }
    ImGui::Image(ImTextureID(app.getFrameTextureId()),
                 {static_cast<float>(size.x), static_cast<float>(size.y - 40)},
                 {0.f, 1.f}, {1.f, 0.f});
    is_render_hovered = ImGui::IsWindowHovered();
    ImGui::End();
}

void lift::ImGuiLayer::onEvent(Event& event) {
    const auto& io = ImGui::GetIO();
    if (io.WantCaptureMouse && !is_render_hovered && event.getEventType() == EventType::MOUSE_MOVE)
        event.handled = true;
}

ivec2 lift::ImGuiLayer::getRenderWindowSize() {
    return render_window_size;
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
        const auto backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}
