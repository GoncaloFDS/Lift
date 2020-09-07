#include "application.h"
#include <core/key_codes.h>
#include <core/profiler.h>
#include <core/timer.h>

#include "algorithm_list.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/texture.h"
#include "assets/uniform_buffer.h"
#include "core/input.h"
#include "effolkronium/random.hpp"
#include "renderer.h"
#include "vulkan/blas.h"
#include "vulkan/depth_buffer.h"
#include "vulkan/graphics_pipeline.h"
#include "vulkan/instance.h"
#include "vulkan/ray_tracing_pipeline.h"
#include "vulkan/render_pass.h"
#include "vulkan/single_time_commands.h"
#include "vulkan/surface.h"
#include "vulkan/swap_chain.h"
#include "vulkan/window.h"

using Random = effolkronium::random_static;

#ifdef NDEBUG
const auto k_validation_layers = std::vector<const char*>();
#else
const auto k_validation_layers = std::vector<const char*> {"VK_LAYER_KHRONOS_validation"};
#endif

Application::Application(const UserSettings& user_settings, const vulkan::WindowData& window_properties)
    : user_settings_(user_settings), is_running_(true) {

    const auto validation_layers = k_validation_layers;

    window_ = std::make_unique<vulkan::Window>(window_properties);
    window_->setEventCallbackFn(LF_BIND_EVENT_FN(Application::onEvent));

    instance_ = std::make_unique<vulkan::Instance>(*window_, validation_layers);
    renderer_ = std::make_unique<Renderer>(*instance_);
}

Application::~Application() {
    scene_.reset();
    deleteSwapChain();
    renderer_.reset();
    instance_.reset();
    window_.reset();
}

void Application::run() {
    Timer::start();

    while (is_running_) {
        window_->poolEvents();

        if (window_->isMinimized())
            continue;

        onUpdate();

        // Render the scene
        auto ubo = getUniformBufferObject(renderer_->swapChain().extent());
        renderer_->beginCommand(*scene_, current_frame_);

        if (!hasReachedTargetFrame()) {
            renderer_->trace(*scene_);
            if (user_settings_.is_denoised) {
                renderer_->denoiseImage();
            }
        }
        else if (user_settings_.denoise_final_image) {
            renderer_->denoiseImage();
        }

        renderer_->display();

        // Update UI
        Statistics stats = {};
        stats.framebuffer_size = window_->framebufferSize();
        stats.frame_rate = static_cast<float>(1 / Timer::delta_time);
        stats.frame_time = Timer::delta_time * 1000.0f;
        stats.denoiser_time = Profiler::getDuration(Profiler::Id::Denoise) * 1000.0f;
        stats.total_samples = total_number_of_samples_;
        stats.total_time = Timer::elapsed_time;
        user_interface_->updateInfo(stats, camera_->state());

        // Render the UI
        renderer_->render(*user_interface_);

        renderer_->endCommand(*scene_, current_frame_, ubo);

        if (!hasReachedTargetFrame()) {
            total_number_of_samples_ += user_settings_.number_of_samples;
            number_of_frames_++;
        } else {
            Timer::stop();
        }

        Timer::tick();
    }

    renderer_->waitDeviceIdle();
}

assets::UniformBufferObject Application::getUniformBufferObject(VkExtent2D extent) const {

    assets::UniformBufferObject ubo = {};
    ubo.modelView = camera_->model_view();
    ubo.projection =
        perspective(radians(user_settings_.fov), extent.width / static_cast<float>(extent.height), 0.1f, 10000.0f);
    ubo.projection[1][1] *= -1;
    // Inverting Y for vulkan,
    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    ubo.model_view_inverse = glm::inverse(ubo.modelView);
    ubo.projection_inverse = glm::inverse(ubo.projection);
    ubo.next_event_estimation = user_settings_.next_event_estimation;
    ubo.aperture = user_settings_.aperture;
    ubo.focus_distance = user_settings_.focus_distance;
    ubo.total_number_of_samples = total_number_of_samples_;
    ubo.number_of_samples = user_settings_.number_of_samples;
    ubo.number_of_bounces = user_settings_.number_of_bounces;
    ubo.seed = Random::get(0u, 1000u);
    ubo.gamma_correction = user_settings_.gamma_correction;
    ubo.tone_map = user_settings_.tone_map;
    ubo.exposure = user_settings_.exposure;
    ubo.debug_normals = user_settings_.debug_normals;
    ubo.debug_radiance = user_settings_.debug_radiance;
    ubo.has_sky = camera_->hasSky();
    ubo.frame = number_of_frames_;

    return ubo;
}

const std::vector<VkExtensionProperties>& Application::extensions() const {
    return instance_->extensions();
}

const std::vector<VkPhysicalDevice>& Application::physicalDevices() const {
    return instance_->physicalDevices();
}

void Application::setPhysicalDevice(VkPhysicalDevice physical_device) {
    renderer_->init(physical_device, *scene_, *instance_);
    loadScene(user_settings_.scene_index);

    renderer_->createAccelerationStructures(*scene_);

    while (window_->isMinimized()) {
        window_->waitForEvents();
    }

    createSwapChain();
}

void Application::createSwapChain() {
    renderer_->createSwapChain(*scene_, static_cast<Algorithm>(user_settings_.algorithm_index));

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

void Application::onUpdate() {
    // Check if the scene has been changed by the user.
    if (scene_index_ != static_cast<uint32_t>(user_settings_.scene_index) ||
        previous_settings_.algorithm_index != user_settings_.algorithm_index) {
        previous_settings_ = user_settings_;
        renderer_->waitDeviceIdle();
        deleteSwapChain();
        renderer_->deleteAccelerationStructures();
        loadScene(user_settings_.scene_index);
        renderer_->createAccelerationStructures(*scene_);
        createSwapChain();
        return;
    }

    bool camera_changed = camera_->onUpdate();
    // Check if the accumulation buffer needs to be reset.
    if (reset_accumulation_ || user_settings_.requiresAccumulationReset(previous_settings_) ||
        !user_settings_.accumulate_rays || camera_changed) {

        total_number_of_samples_ = 0;
        number_of_frames_ = 0;
        reset_accumulation_ = false;
        Timer::restart();
    }

    if (!user_settings_.is_denoised) {
        Profiler::reset(Profiler::Id::Denoise);
    }

    previous_settings_ = user_settings_;

    camera_->setMoveSpeed(user_settings_.camera_move_speed);
    camera_->setMouseSpeed(user_settings_.camera_mouse_speed);
}

void Application::loadScene(const uint32_t scene_index) {
    Profiler profiler("Scene loading took");
    auto assets = SceneList::all_scenes[scene_index].second();

    LF_INFO("Loading Scene {0}", SceneList::all_scenes[scene_index].first.c_str());

    if (assets.textures.empty()) {
        assets.textures.push_back(assets::Texture::loadTexture("../resources/textures/white.png"));
    }

    scene_ = std::make_unique<assets::Scene>(renderer_->commandPool(), assets);
    scene_index_ = scene_index;

    camera_initial_state_ = assets.camera;
    user_settings_.fov = camera_initial_state_.field_of_view;
    user_settings_.aperture = camera_initial_state_.aperture;
    user_settings_.focus_distance = camera_initial_state_.focus_distance;
    user_settings_.gamma_correction = camera_initial_state_.gamma_correction;
    user_settings_.camera_move_speed = camera_initial_state_.move_speed;
    user_settings_.camera_mouse_speed = camera_initial_state_.look_speed;

    camera_initial_state_.aspect_ratio = float(window_->framebufferSize().width) / window_->framebufferSize().height;
    camera_ = std::make_unique<Camera>(camera_initial_state_);

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
        createSwapChain();
        LF_INFO(e.toString());
    }
    return false;
}

bool Application::onWindowMinimize(WindowMinimizeEvent& e) {
    return false;
}

bool Application::onMouseMove(MouseMovedEvent& e) {
    if (user_interface_->wantsToCaptureMouse()) {
        return true;
    }
    Input::moveMouse(e.x(), e.y());

    if (Input::isKeyPressed(LF_MOUSE_BUTTON_1)) {
        const auto delta = Input::mouseDelta();
        const auto window_size = window_->framebufferSize();
        camera_->orbit(delta.x / float(window_size.width), delta.y / float(window_size.height));
        reset_accumulation_ = true;
    }
    return false;
}

bool Application::onMouseScroll(MouseScrolledEvent& e) {
    return false;
}

bool Application::onKeyPress(KeyPressedEvent& e) {
    if (user_interface_->wantsToCaptureKeyboard()) {
        return true;
    }

    if (e.keyCode() == LF_KEY_ESCAPE) {
        window_->close();
    }
    if (e.keyCode() == LF_KEY_F1) {
        user_settings_.show_settings = !user_settings_.show_settings;
    }
    if (e.keyCode() == LF_KEY_F2) {
        user_settings_.show_overlay = !user_settings_.show_overlay;
    }
    if (e.keyCode() == LF_KEY_F3) {
        user_settings_.is_denoised = !user_settings_.is_denoised;
    }
    if (e.keyCode() == LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD);
    }
    if (e.keyCode() == LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK);
    }
    if (e.keyCode() == LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT);
    }
    if (e.keyCode() == LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT);
    }
    if (e.keyCode() == LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP);
    }
    if (e.keyCode() == LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN);
    }

    return false;
}

bool Application::onKeyRelease(KeyReleasedEvent& e) {
    if (user_interface_->wantsToCaptureKeyboard()) {
        return true;
    }

    if (e.keyCode() == LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD, -1.0f);
    }
    if (e.keyCode() == LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK, -1.0f);
    }
    if (e.keyCode() == LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT, -1.0f);
    }
    if (e.keyCode() == LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT, -1.0f);
    }
    if (e.keyCode() == LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP, -1.0f);
    }
    if (e.keyCode() == LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN, -1.0f);
    }
    return false;
}
bool Application::hasReachedTargetFrame() const {
    bool reached_frame = user_settings_.target_frame_count != 0 && total_number_of_samples_ >= user_settings_.target_frame_count;
    bool reached_time = user_settings_.target_render_time > 0.0f && Timer::elapsed_time >= user_settings_.target_render_time;
    return reached_frame || reached_time;
}
