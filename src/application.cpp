#include <pch.h>
#include "application.h"

#include "vulkan/window.h"
#include "vulkan/swap_chain.h"
#include "vulkan/device.h"
#include "assets/texture.h"
#include "user_settings.h"
#include "imgui/imgui_layer.h"
#include "vulkan/single_time_commands.h"
#include "vulkan/pipeline_layout.h"
#include "vulkan/image_view.h"
#include "vulkan/image.h"
#include "vulkan/buffer.h"
#include "vulkan/tlas.h"
#include "vulkan/shader_binding_table.h"
#include "vulkan/ray_tracing_pipeline.h"
#include "vulkan/device_procedures.h"
#include "vulkan/blas.h"
#include "vulkan/depth_buffer.h"
#include "vulkan/fence.h"
#include "vulkan/graphics_pipeline.h"
#include "vulkan/instance.h"
#include "vulkan/render_pass.h"
#include "vulkan/semaphore.h"
#include "vulkan/surface.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/camera.h"
#include "assets/uniform_buffer.h"
#include "core/input.h"
#include "renderer.h"
#include "denoiser/denoiser_optix.h"

namespace lift {
using namespace vulkan;

#ifdef NDEBUG
const auto k_validation_layers = std::vector<const char*>();
#else
const auto k_validation_layers = std::vector<const char*>{"VK_LAYER_KHRONOS_validation"};
#endif

Application::Application(const UserSettings& user_settings, const WindowData& window_properties, const bool vsync) :
    user_settings_(user_settings),
    vsync_(vsync),
    is_running_(true) {

    const auto validation_layers = k_validation_layers;

    window_ = std::make_unique<Window>(window_properties);
    window_->setEventCallbackFn(LF_BIND_EVENT_FN(Application::onEvent));

    instance_ = std::make_unique<Instance>(*window_, validation_layers);
    renderer_ = std::make_unique<Renderer>(*instance_, vsync);
    denoiser_ = std::make_unique<DenoiserOptix>();
    denoiser_->initOptix();
}

Application::~Application() {
    scene_.reset();
    deleteSwapChain();
    renderer_.reset();
    instance_.reset();
    window_.reset();
}

void Application::run() {
    current_frame_ = 0;

    while (is_running_) {
        window_->poolEvents();

        if (window_->isMinimized())
            continue;

        const auto prev_time = time_;
        time_ = window_->time();
        const auto delta_time = time_ - prev_time;

        onUpdate();

        checkAndUpdateBenchmarkState(prev_time);

        // Render the scene
        auto ubo = getUniformBufferObject(renderer_->swapChain().extent());
        renderer_->beginCommand(*scene_, current_frame_);

        if (user_settings_.isRayTraced) {
            renderer_->trace(*scene_);
        } else {
            renderer_->render(*scene_);
        }

        // Render the UI
        Statistics stats = {};
        stats.framebufferSize = window_->framebufferSize();
        stats.frameRate = static_cast<float>(1 / delta_time);

        if (user_settings_.isRayTraced) {
            const auto extent = renderer_->swapChain().extent();

            stats.rayRate = float(extent.width * extent.height * number_of_samples_ / (delta_time * 1000000000));
            stats.totalSamples = total_number_of_samples_;
        }

        renderer_->render(*user_interface_, stats);

        renderer_->endCommand(*scene_, current_frame_, ubo);

    }

    renderer_->waitDeviceIdle();
}

assets::UniformBufferObject Application::getUniformBufferObject(VkExtent2D extent) const {
    const auto camera_rot_x = static_cast<float>(camera_y_ / 300.0);
    const auto camera_rot_y = static_cast<float>(camera_x_ / 300.0);

    const auto& init = camera_initial_state_;
    const auto view = init.modelView;
    const auto model = glm::rotate(mat4(1.0f), camera_rot_y * radians(90.0f), vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(mat4(1.0f), camera_rot_x * radians(90.0f), vec3(1.0f, 0.0f, 0.0f));

    assets::UniformBufferObject ubo = {};
    ubo.modelView = view * model;
    ubo.projection = perspective(radians(user_settings_.fieldOfView),
                                 extent.width / static_cast<float>(extent.height),
                                 0.1f,
                                 10000.0f);
    ubo.projection[1][1] *=
        -1; // Inverting Y for vulkan, https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    ubo.modelViewInverse = glm::inverse(ubo.modelView);
    ubo.projectionInverse = glm::inverse(ubo.projection);
    ubo.aperture = user_settings_.aperture;
    ubo.focusDistance = user_settings_.focusDistance;
    ubo.totalNumberOfSamples = total_number_of_samples_;
    ubo.numberOfSamples = number_of_samples_;
    ubo.numberOfBounces = user_settings_.numberOfBounces;
    ubo.randomSeed = 1;
    ubo.gammaCorrection = user_settings_.gammaCorrection;
    ubo.hasSky = init.hasSky;

    return ubo;
}

const std::vector<VkExtensionProperties>& Application::extensions() const {
    return instance_->extensions();
}

const std::vector<VkPhysicalDevice>& Application::physicalDevices() const {
    return instance_->physicalDevices();
}

void Application::setPhysicalDevice(VkPhysicalDevice physical_device) {
    renderer_->init(physical_device, *scene_);
    loadScene(user_settings_.sceneIndex);

    renderer_->createAccelerationStructures(*scene_);

    while (window_->isMinimized()) {
        window_->waitForEvents();
    }

    createSwapChain();
}

void Application::createSwapChain() {
    renderer_->createSwapChain(*scene_);

    user_interface_ = std::make_unique<ImguiLayer>(renderer_->commandPool(),
                                                   renderer_->swapChain(),
                                                   renderer_->depthBuffer(),
                                                   user_settings_);
    reset_accumulation_ = true;
}

void Application::deleteSwapChain() {
    user_interface_.reset();
    renderer_->deleteSwapChain();
}

void lift::Application::onUpdate() {
    // Check if the scene has been changed by the user.
    if (scene_index_ != static_cast<uint32_t>(user_settings_.sceneIndex)) {
        renderer_->waitDeviceIdle();
        deleteSwapChain();
        renderer_->deleteAccelerationStructures();
        loadScene(user_settings_.sceneIndex);
        renderer_->createAccelerationStructures(*scene_);
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

    number_of_samples_ = clamp(user_settings_.maxNumberOfSamples - total_number_of_samples_,
                               0u, user_settings_.numberOfSamples);
    total_number_of_samples_ += number_of_samples_;

}

void Application::loadScene(const uint32_t scene_index) {
    auto[models, textures] = SceneList::allScenes[scene_index].second(camera_initial_state_);

    LF_WARN("Loading Scene {0}", SceneList::allScenes[scene_index].first.c_str());

    if (textures.empty()) {
        textures.push_back(assets::Texture::loadTexture("../resources/textures/white.png", SamplerConfig()));
    }

    scene_ = std::make_unique<assets::Scene>(renderer_->commandPool(),
                                             std::move(models),
                                             std::move(textures),
                                             true);
    scene_index_ = scene_index;

    user_settings_.fieldOfView = camera_initial_state_.fieldOfView;
    user_settings_.aperture = camera_initial_state_.aperture;
    user_settings_.focusDistance = camera_initial_state_.focusDistance;
    user_settings_.gammaCorrection = camera_initial_state_.gammaCorrection;

    camera_x_ = 0;
    camera_y_ = 0;

    period_total_frames_ = 0;
    reset_accumulation_ = true;
}

void Application::onEvent(Event& event) {
    EventDispatcher dispatcher(event);
    dispatcher.dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::onWindowMinimize));

    dispatcher.dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::onMouseMove));
    dispatcher.dispatch<MouseScrolledEvent>(LF_BIND_EVENT_FN(Application::onMouseScroll));
    dispatcher.dispatch<KeyPressedEvent>(LF_BIND_EVENT_FN(Application::onKeyPress));
    dispatcher.dispatch<KeyReleasedEvent>(LF_BIND_EVENT_FN(Application::onKeyRelease));

}

bool Application::onWindowClose(WindowCloseEvent& e) {
    is_running_ = false;
    LF_INFO(e.toString());
    return false;
}

bool Application::onWindowResize(WindowResizeEvent& e) {
    if (e.height() && e.width()) {
        renderer_->waitDeviceIdle();
        deleteSwapChain();
        renderer_->deleteAccelerationStructures();
        loadScene(user_settings_.sceneIndex);
        renderer_->createAccelerationStructures(*scene_);
        createSwapChain();
        LF_INFO(e.toString());
    }
    return false;
}

bool Application::onWindowMinimize(WindowMinimizeEvent& e) {

    return false;
}

bool Application::onMouseMove(MouseMovedEvent& e) {
    if (Input::isKeyPressed(LF_MOUSE_BUTTON_1)) {
        camera_x_ += static_cast<float>(e.x() - mouse_x_);
        camera_y_ += static_cast<float>(e.y() - mouse_y_);

        reset_accumulation_ = true;
    }
    mouse_x_ = e.x();
    mouse_y_ = e.y();
    return false;
}

bool Application::onMouseScroll(MouseScrolledEvent& e) {
    return false;
}

bool Application::onKeyPress(KeyPressedEvent& e) {
    if (user_interface_->wantsToCaptureKeyboard()) {
        return true;
    }
    switch (e.keyCode()) {
        case LF_KEY_ESCAPE:
            window_->close();
            break;
        case LF_KEY_F1:
            user_settings_.showSettings = !user_settings_.showSettings;
            break;
        case LF_KEY_F2:
            user_settings_.showOverlay = !user_settings_.showOverlay;
            break;
        case LF_KEY_F3:
            user_settings_.isRayTraced = !user_settings_.isRayTraced;
            break;
        default:
            break;

    }
    return false;
}

bool Application::onKeyRelease(KeyReleasedEvent& e) {
    return false;
}

void lift::Application::checkAndUpdateBenchmarkState(double prev_time) {
    if (!user_settings_.benchmark) {
        return;
    }

    // Initialise scene benchmark timers
    if (period_total_frames_ == 0) {
        std::cout << std::endl;
        std::cout << "Benchmark: Start scene #" << scene_index_ << " '" << SceneList::allScenes[scene_index_].first
                  << "'"
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
            period_total_frames_ != 0 && window_->time() - scene_initial_time_ > user_settings_.benchmarkMaxTime;
        const bool sample_limit_reached = number_of_samples_ == 0;

        if (time_limit_reached || sample_limit_reached) {
            if (!user_settings_.benchmarkNextScenes
                || static_cast<size_t>(user_settings_.sceneIndex) == SceneList::allScenes.size() - 1) {
                window_->close();
            }

            std::cout << std::endl;
            user_settings_.sceneIndex += 1;
        }
    }
}
}
