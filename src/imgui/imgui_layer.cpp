#include "imgui_layer.h"
#include "scene_list.h"
#include "user_settings.h"
#include "platform/vulkan/descriptor_pool.h"
#include "platform/vulkan/device.h"
#include "platform/vulkan/frame_buffer.h"
#include "platform/vulkan/instance.h"
#include "platform/vulkan/render_pass.h"
#include "platform/vulkan/single_time_commands.h"
#include "platform/vulkan/surface.h"
#include "platform/vulkan/swap_chain.h"
#include "platform/vulkan/window.h"

#include "imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include <array>
#include <memory>

namespace {
void checkVulkanResultCallback(const VkResult err) {
    if (err != VK_SUCCESS) {
//			Throw(std::runtime_error(std::string("ImGui vulkan error (") + vulkan::ToString(err) + ")"));
    }
}
}

ImguiLayer::ImguiLayer(vulkan::CommandPool& command_pool,
                       const vulkan::SwapChain& swap_chain,
                       const vulkan::DepthBuffer& depth_buffer,
                       UserSettings& user_settings) :
    user_settings_(user_settings) {
    const auto& device = swap_chain.device();
    const auto& window = device.surface().instance().window();

    // Initialise descriptor pool and render pass for ImGui.
    const std::vector<vulkan::DescriptorBinding> descriptor_bindings = {
        {0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0},
    };
    descriptor_pool_ = std::make_unique<vulkan::DescriptorPool>(device, descriptor_bindings, 1);
    render_pass_ = std::make_unique<vulkan::RenderPass>(swap_chain, depth_buffer, false, false);

    // Initialise ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Initialise ImGui GLFW adapter
    if (!ImGui_ImplGlfw_InitForVulkan(window.handle(), false)) {
//		Throw(std::runtime_error("failed to initialise ImGui GLFW adapter"));
    }

    // Initialise ImGui vulkan adapter
    ImGui_ImplVulkan_InitInfo vulkan_init = {};
    vulkan_init.Instance = device.surface().instance().Handle();
    vulkan_init.PhysicalDevice = device.physicalDevice();
    vulkan_init.Device = device.Handle();
    vulkan_init.QueueFamily = device.graphicsFamilyIndex();
    vulkan_init.Queue = device.graphicsQueue();
    vulkan_init.PipelineCache = nullptr;
    vulkan_init.DescriptorPool = descriptor_pool_->Handle();
    vulkan_init.MinImageCount = swap_chain.minImageCount();
    vulkan_init.ImageCount = static_cast<uint32_t>(swap_chain.images().size());
    vulkan_init.Allocator = nullptr;
    vulkan_init.CheckVkResultFn = checkVulkanResultCallback;

    if (!ImGui_ImplVulkan_Init(&vulkan_init, render_pass_->Handle())) {
//		Throw(std::runtime_error("failed to initialise ImGui vulkan adapter"));
    }

    auto& io = ImGui::GetIO();

    // No ini file.
    io.IniFilename = nullptr;

    // Window scaling and style.
    const auto scale_factor = window.contentScale();

    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(scale_factor);

    // Upload ImGui fonts
    if (!io.Fonts->AddFontFromFileTTF("../resources/fonts/Cousine-Regular.ttf", 13 * scale_factor)) {
//		Throw(std::runtime_error("failed to load ImGui font"));
    }

    vulkan::SingleTimeCommands::submit(command_pool, [](VkCommandBuffer command_buffer) {
        if (!ImGui_ImplVulkan_CreateFontsTexture(command_buffer)) {
//			Throw(std::runtime_error("failed to create ImGui font textures"));
        }
    });

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

ImguiLayer::~ImguiLayer() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImguiLayer::render(VkCommandBuffer command_buffer,
                        const vulkan::FrameBuffer& frame_buffer,
                        const Statistics& statistics) {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    drawSettings();
    drawOverlay(statistics);
    //ImGui::ShowStyleEditor();
    ImGui::Render();

    std::array<VkClearValue, 2> clear_values = {};
    clear_values[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_->Handle();
    render_pass_info.framebuffer = frame_buffer.Handle();
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = render_pass_->swapChain().extent();
    render_pass_info.clearValueCount = 0;// static_cast<uint32_t>(clearValues.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
    vkCmdEndRenderPass(command_buffer);
}

bool ImguiLayer::wantsToCaptureKeyboard() {
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImguiLayer::wantsToCaptureMouse() {
    return ImGui::GetIO().WantCaptureMouse;
}

void ImguiLayer::drawSettings() {
    if (!settings().showSettings) {
        return;
    }

    const float distance = 10.0f;
    const ImVec2 pos = ImVec2(distance, distance);
    const ImVec2 posPivot = ImVec2(0.0f, 0.0f);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, posPivot);

    const auto flags = ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("Settings", &settings().showSettings, flags)) {
        std::vector<const char*> scenes;
        scenes.reserve(SceneList::allScenes.size());
        for (const auto& scene : SceneList::allScenes) {
            scenes.push_back(scene.first.c_str());
        }

        ImGui::Text("Help");
        ImGui::Separator();
        ImGui::BulletText("Press F1 to toggle Settings.");
        ImGui::BulletText("Press F2 to toggle Statistics.");
        ImGui::BulletText("Press R to toggle ray tracing.");
        ImGui::NewLine();

        ImGui::Text("Scene");
        ImGui::Separator();
        ImGui::PushItemWidth(-1);
        ImGui::Combo("", &settings().sceneIndex, scenes.data(), static_cast<int>(scenes.size()));
        ImGui::PopItemWidth();
        ImGui::Checkbox("Show statistics overlay", &settings().showOverlay);
        ImGui::NewLine();

        ImGui::Text("Ray Tracing");
        ImGui::Separator();
        ImGui::Checkbox("Enable ray tracing", &settings().isRayTraced);
        ImGui::Checkbox("Accumulate rays between frames", &settings().accumulateRays);
        uint32_t min = 1, max = 128;
        ImGui::SliderScalar("Samples", ImGuiDataType_U32, &settings().numberOfSamples, &min, &max);
        min = 1, max = 32;
        ImGui::SliderScalar("Bounces", ImGuiDataType_U32, &settings().numberOfBounces, &min, &max);
        ImGui::NewLine();

        ImGui::Text("Camera");
        ImGui::Separator();
        ImGui::SliderFloat("FoV", &settings().fieldOfView, 10.0f, 90.0f, "%.0f");
        ImGui::SliderFloat("Aperture", &settings().aperture, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Focus", &settings().focusDistance, 0.1f, 20.0f, "%.1f");
        ImGui::Checkbox("Apply gamma correction", &settings().gammaCorrection);
        ImGui::NewLine();
    }
    ImGui::End();
}

void ImguiLayer::drawOverlay(const Statistics& statistics) {
    if (!settings().showOverlay) {
        return;
    }

    const auto& io = ImGui::GetIO();
    const float distance = 10.0f;
    const ImVec2 pos = ImVec2(io.DisplaySize.x - distance, distance);
    const ImVec2 pos_pivot = ImVec2(1.0f, 0.0f);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.3f); // Transparent background

    const auto flags = ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("Statistics", &settings().showOverlay, flags)) {
        ImGui::Text("Statistics (%dx%d):", statistics.framebufferSize.width, statistics.framebufferSize.height);
        ImGui::Separator();
        ImGui::Text("Frame rate: %.1f fps", statistics.frameRate);
        ImGui::Text("Primary ray rate: %.2f Gr/s", statistics.rayRate);
        ImGui::Text("Accumulated samples:  %u", statistics.totalSamples);
    }
    ImGui::End();
}
