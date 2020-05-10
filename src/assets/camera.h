#pragma once

#include "glm.h"

using namespace glm;

enum class Direction { UP, DOWN, RIGHT, LEFT, FORWARD, BACK };

struct CameraState {
    glm::vec3 eye;
    glm::vec3 look_at;
    glm::vec3 up;
    float field_of_view;
    float aperture;
    float focus_distance;
    float aspect_ratio;
    bool gamma_correction;
    bool has_sky;
    float move_speed {200.0f};
    float look_speed {80.0f};
};

class Camera {
public:
    Camera(const CameraState& s);

    [[nodiscard]] mat4 model_view() const { return glm::lookAt(eye(), lookAt(), up()); }
    [[nodiscard]] vec3 direction() const { return normalize(state_.look_at - state_.eye); }
    [[nodiscard]] const vec3& eye() const { return state_.eye; };
    [[nodiscard]] const vec3& lookAt() const { return state_.look_at; }
    [[nodiscard]] const vec3& up() const { return state_.up; }
    [[nodiscard]] float fovy() const { return state_.field_of_view; }
    [[nodiscard]] float aspectRatio() const { return state_.aspect_ratio; }
    [[nodiscard]] bool hasSky() const { return state_.has_sky; }
    [[nodiscard]] CameraState state() const { return state_; }

    void setDirection(const vec3& direction) { state_.look_at = state_.eye + length(state_.look_at - state_.eye); }
    void setEye(const vec3& eye) {
        state_.eye = eye;
        changed_ = true;
    }
    void setLookAt(const vec3& look_at) {
        state_.look_at = look_at;
        changed_ = true;
    }
    void setUp(const vec3& up) {
        state_.up = up;
        changed_ = true;
    }
    void setFovy(const float fovy) {
        state_.field_of_view = fovy;
        changed_ = true;
    }
    void setAspectRatio(const float& aspect_ratio) {
        state_.aspect_ratio = aspect_ratio;
        changed_ = true;
    }
    void setMoveSpeed(const float amount) { state_.move_speed = amount; }
    void setMouseSpeed(const float amount) { state_.look_speed = amount; }
    void setMoveDirection(enum Direction direction, float amount = 1.0f);

    bool onUpdate();

    void orbit(float dx, float dy);

private:
    CameraState state_;

    vec3 vector_u_ {1.0f};
    vec3 vector_v_ {1.0f};
    vec3 vector_w_ {1.0f};

    vec3 norm_vector_u_ {1.0f};
    vec3 norm_vector_v_ {1.0f};
    vec3 norm_vector_w_ {1.0f};

    vec3 move_dir_ {0.0f};

    bool changed_ = true;

    void move();
};
