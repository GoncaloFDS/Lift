#include <core/timer.h>

#include "camera.h"
#include <algorithm>

Camera::Camera() :
    eye_(0.0f, 0.0f, -12.f), look_at_(0.0f, 0.0f, 0.0f), up_(0.0f, 1.0f, 0.0f), aspect_ratio_(1.0f),
    field_of_view_(36.0f), changed_(true) {
}

Camera::Camera(const CameraState& s) :
    eye_(s.eye), look_at_(s.look_at), up_(s.up), aspect_ratio_(s.aspect_ratio), field_of_view_(s.field_of_view),
    has_sky(s.has_sky), changed_(true) {
}

bool Camera::onUpdate() {
    move();

    if (!changed_) {
        return false;
    }

    vector_w_ = look_at_ - eye_;  // Do not normalize -- it implies focal length
    const auto w_length = length(vector_w_);
    vector_u_ = normalize(cross(vector_w_, up_));
    vector_v_ = normalize(cross(vector_u_, vector_w_));

    norm_vector_u_ = vector_u_;
    norm_vector_v_ = vector_v_;
    norm_vector_w_ = normalize(vector_w_);

    const auto v_length = w_length * tanf(0.5f * field_of_view_ * pi<float>() / 180.f);
    vector_v_ *= v_length;
    const auto u_length = v_length * aspect_ratio_;
    vector_u_ *= u_length;
    changed_ = false;
    return true;
}

void Camera::orbit(const float dx, const float dy) {
    vec3 t = look_at_ - eye_;
    if (fabs(dot(normalize(t), vec3(0, 1, 0))) < 0.999f || t.y * dy < 0) {
        t = rotate(mat4(1.0f), dx * mouse_look_speed_ * Timer::deltaTime, norm_vector_v_)
            * rotate(mat4(1.0f), dy * mouse_look_speed_ * Timer::deltaTime, norm_vector_u_) * vec4(t, 1);
        look_at_ = eye_ + t;
        changed_ = true;
    }
}

void Camera::strafe(const float dx, const float dy) {
    const auto mat = translate(mat4(1.0f), dx * mouse_strafe_speed_ * Timer::deltaTime * norm_vector_u_)
                     * translate(mat4(1.0f), dy * mouse_strafe_speed_ * Timer::deltaTime * norm_vector_v_);
    eye_ = mat * vec4(eye_, 1.0f);
    look_at_ = mat * vec4(look_at_, 1.0f);
    changed_ = true;
}

void Camera::zoom(const float amount) {
    field_of_view_ += amount * mouse_zoom_speed_ * Timer::deltaTime;
    field_of_view_ = std::clamp(field_of_view_, 0.001f, 180.0f);
    changed_ = true;
}

void Camera::setMoveDirection(enum Direction direction, float amount) {
    switch (direction) {
        case Direction::UP:
            move_dir_ += vec3(0.0f, 1.0f, 0.0f) * amount;
            break;
        case Direction::DOWN:
            move_dir_ -= vec3(0.0f, 1.0f, 0.0f) * amount;
            break;
        case Direction::RIGHT:
            move_dir_ += vec3(1.0f, 0.0f, 0.0f) * amount;
            break;
        case Direction::LEFT:
            move_dir_ -= vec3(1.0f, 0.0f, 0.0f) * amount;
            break;
        case Direction::FORWARD:
            move_dir_ += vec3(0.0f, 0.0f, 1.0f) * amount;
            break;
        case Direction::BACK:
            move_dir_ -= vec3(0.0f, 0.0f, 1.0f) * amount;
            break;
    }
    move_dir_ = clamp(move_dir_, vec3(-1.0f), vec3(1.0f));
}

void Camera::move() {
    if (move_dir_ != vec3(0.0f)) {
        vec3 dir = move_dir_.x * norm_vector_u_ + move_dir_.y * norm_vector_v_ + move_dir_.z * norm_vector_w_;
        const auto mat = translate(mat4(1.0f), camera_move_speed_ * Timer::deltaTime * dir);
        eye_ = mat * vec4(eye_, 1.0f);
        look_at_ = mat * vec4(look_at_, 1.0f);
        changed_ = true;
    }
}
