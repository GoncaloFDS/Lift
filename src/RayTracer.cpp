#include "RayTracer.h"
#include "imgui_layer.h"
#include "user_settings.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/texture.h"
#include "assets/uniform_buffer.h"
#include "vulkan/device.h"
#include "vulkan/swap_chain.h"
#include "vulkan/window.h"
#include <iostream>
#include <memory>
#include <sstream>

namespace {
const bool EnableValidationLayers =
#ifdef NDEBUG
    false;
#else
true;
#endif
}

RayTracer::RayTracer(const UserSettings& user_settings,
                     const vulkan::WindowProperties& window_properties,
                     const bool vsync)
    : Application(window_properties, vsync, EnableValidationLayers), user_settings_(user_settings) {
    checkFramebufferSize();
}

RayTracer::~RayTracer() {
    scene_.reset();
}

assets::UniformBufferObject RayTracer::getUniformBufferObject(VkExtent2D extent) const {
    const auto camera_rot_x = static_cast<float>(camera_y_ / 300.0);
    const auto camera_rot_y = static_cast<float>(camera_x_ / 300.0);

    const auto& init = camera_initial_sate_;
    const auto view = init.modelView;
    const auto model = glm::rotate(glm::mat4(1.0f), camera_rot_y * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), camera_rot_x * glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    assets::UniformBufferObject ubo = {};
    ubo.ModelView = view * model;
    ubo.Projection = glm::perspective(glm::radians(user_settings_.fieldOfView),
                                      extent.width / static_cast<float>(extent.height),
                                      0.1f,
                                      10000.0f);
    ubo.Projection[1][1] *=
        -1; // Inverting Y for vulkan, https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    ubo.ModelViewInverse = glm::inverse(ubo.ModelView);
    ubo.ProjectionInverse = glm::inverse(ubo.Projection);
    ubo.Aperture = user_settings_.aperture;
    ubo.FocusDistance = user_settings_.focusDistance;
    ubo.TotalNumberOfSamples = total_number_of_samples_;
    ubo.NumberOfSamples = number_of_samples_;
    ubo.NumberOfBounces = user_settings_.numberOfBounces;
    ubo.RandomSeed = 1;
    ubo.GammaCorrection = user_settings_.gammaCorrection;
    ubo.HasSky = init.hasSky;

    return ubo;
}

void RayTracer::onDeviceSet() {
    Application::onDeviceSet();

    loadScene(user_settings_.sceneIndex);
    createAccelerationStructures();
}

void RayTracer::createSwapChain() {
    Application::createSwapChain();

    user_interface_ = std::make_unique<ImguiLayer>(commandPool(), swapChain(), depthBuffer(), user_settings_);
    reset_accumulation_ = true;

    checkFramebufferSize();
}

void RayTracer::deleteSwapChain() {
    user_interface_.reset();

    Application::deleteSwapChain();
}

void RayTracer::drawFrame() {
    // Check if the scene has been changed by the user.
    if (scene_index_ != static_cast<uint32_t>(user_settings_.sceneIndex)) {
        device().waitIdle();
        deleteSwapChain();
        deleteAccelerationStructures();
        loadScene(user_settings_.sceneIndex);
        createAccelerationStructures();
        createSwapChain();
        return;
    }

    // Check if the accumulation buffer needs to be reset.
    if (reset_accumulation_ ||
        user_settings_.requiresAccumulationReset(previous_settings_) ||
        !user_settings_.accumulateRays) {
        total_number_of_samples_ = 0;
        reset_accumulation_ = false;
    }

    previous_settings_ = user_settings_;

    // Keep track of our sample count.
    number_of_samples_ =
        glm::clamp(user_settings_.maxNumberOfSamples - total_number_of_samples_, 0u, user_settings_.numberOfSamples);
    total_number_of_samples_ += number_of_samples_;

    Application::drawFrame();
}

void RayTracer::render(VkCommandBuffer command_buffer, uint32_t image_index) {
    // Record delta time between calls to Render.
    const auto prev_time = time_;
    time_ = window().time();
    const auto delta_time = time_ - prev_time;

    // Check the current state of the benchmark, update it for the new frame.
    checkAndUpdateBenchmarkState(prev_time);

    // Render the scene
    vulkan::ray_tracing::Application::render(command_buffer, image_index);

    // Render the UI
    Statistics stats = {};
    stats.framebufferSize = window().framebufferSize();
    stats.frameRate = static_cast<float>(1 / delta_time);

    if (user_settings_.isRayTraced) {
        const auto extent = swapChain().extent();

        stats.rayRate = static_cast<float>( double(extent.width * extent.height) * number_of_samples_ / (delta_time * 1000000000));
        stats.totalSamples = total_number_of_samples_;
    }

    user_interface_->render(command_buffer, swapChainFrameBuffer(image_index), stats);
}

void RayTracer::onKey(int key, int scan_code, int action, int mods) {
    if (user_interface_->wantsToCaptureKeyboard()) {
        return;
    }

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE: window().close();
                break;
            default: break;
        }

        if (!user_settings_.benchmark) {
            switch (key) {
                case GLFW_KEY_F1: user_settings_.showSettings = !user_settings_.showSettings;
                    break;
                case GLFW_KEY_F2: user_settings_.showOverlay = !user_settings_.showOverlay;
                    break;
                case GLFW_KEY_R: user_settings_.isRayTraced = !user_settings_.isRayTraced;
                    break;
                case GLFW_KEY_W: is_wire_frame_ = !is_wire_frame_;
                    break;
                default: break;
            }
        }
    }
}

void RayTracer::onCursorPosition(const double xpos, const double ypos) {
    if (user_settings_.benchmark ||
        user_interface_->wantsToCaptureKeyboard() ||
        user_interface_->wantsToCaptureMouse()) {
        return;
    }

    if (mouse_left_pressed_) {
        const auto delta_x = static_cast<float>(xpos - mouse_x_);
        const auto delta_y = static_cast<float>(ypos - mouse_y_);

        camera_x_ += delta_x;
        camera_y_ += delta_y;

        reset_accumulation_ = true;
    }

    mouse_x_ = xpos;
    mouse_y_ = ypos;
}

void RayTracer::onMouseButton(const int button, const int action, const int mods) {
    if (user_settings_.benchmark ||
        user_interface_->wantsToCaptureMouse()) {
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouse_left_pressed_ = action == GLFW_PRESS;
    }
}

void RayTracer::loadScene(const uint32_t scene_index) {
    auto[models, textures] = SceneList::allScenes[scene_index].second(camera_initial_sate_);

    // If there are no texture, add a dummy one. It makes the pipeline setup a lot easier.
    if (textures.empty()) {
        textures.push_back(assets::Texture::LoadTexture("../resources/textures/white.png", vulkan::SamplerConfig()));
    }

    scene_ = std::make_unique<assets::Scene>(commandPool(), std::move(models), std::move(textures), true);
    scene_index_ = scene_index;

    user_settings_.fieldOfView = camera_initial_sate_.fieldOfView;
    user_settings_.aperture = camera_initial_sate_.aperture;
    user_settings_.focusDistance = camera_initial_sate_.focusDistance;
    user_settings_.gammaCorrection = camera_initial_sate_.gammaCorrection;

    camera_x_ = 0;
    camera_y_ = 0;

    period_total_frames_ = 0;
    reset_accumulation_ = true;
}

void RayTracer::checkAndUpdateBenchmarkState(double prev_time) {
    if (!user_settings_.benchmark) {
        return;
    }

    // Initialise scene benchmark timers
    if (period_total_frames_ == 0) {
        std::cout << std::endl;
        std::cout << "Benchmark: Start scene #" << scene_index_ << " '" << SceneList::allScenes[scene_index_].first << "'"
                  << std::endl;
        scene_initial_time_ = time_;
        period_initial_time_ = time_;
    }

    // Print out the frame rate at regular intervals.
    {
        const double period = 5;
        const double prev_total_time = prev_time - period_initial_time_;
        const double total_time = time_ - period_initial_time_;

        if (period_total_frames_ != 0
            && static_cast<uint64_t>(prev_total_time / period) != static_cast<uint64_t>(total_time / period)) {
            std::cout << "Benchmark: " << period_total_frames_ / total_time << " fps" << std::endl;
            period_initial_time_ = time_;
            period_total_frames_ = 0;
        }

        period_total_frames_++;
    }

    // If in benchmark mode, bail out from the scene if we've reached the time or sample limit.
    {
        const bool time_limit_reached =
            period_total_frames_ != 0 && window().time() - scene_initial_time_ > user_settings_.benchmarkMaxTime;
        const bool sample_limit_reached = number_of_samples_ == 0;

        if (time_limit_reached || sample_limit_reached) {
            if (!user_settings_.benchmarkNextScenes
                || static_cast<size_t>(user_settings_.sceneIndex) == SceneList::allScenes.size() - 1) {
                window().close();
            }

            std::cout << std::endl;
            user_settings_.sceneIndex += 1;
        }
    }
}

void RayTracer::checkFramebufferSize() const {
    // Check the framebuffer size when requesting a fullscreen window, as it's not guaranteed to match.
    const auto& cfg = window().config();
    const auto fb_size = window().framebufferSize();

    if (user_settings_.benchmark && cfg.fullscreen && (fb_size.width != cfg.width || fb_size.height != cfg.height)) {
        std::ostringstream out;
        out << "framebuffer fullscreen size mismatch (requested: ";
        out << cfg.width << "x" << cfg.height;
        out << ", got: ";
        out << fb_size.width << "x" << fb_size.height << ")";

//		Throw(std::runtime_error(out.str()));
    }
}
