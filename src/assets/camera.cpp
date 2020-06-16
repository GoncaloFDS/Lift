#include <core/timer.h>

#include "camera.h"

Camera::Camera(const CameraState& s) {
   state_ = s;
   changed_ = true;
}

bool Camera::onUpdate() {
    move();

    if (!changed_) {
        return false;
    }

    vector_w_ = state_.look_at - state_.eye;  // Do not normalize -- it implies focal length
    const auto w_length = length(vector_w_);
    vector_u_ = normalize(cross(vector_w_, state_.up));
    vector_v_ = normalize(cross(vector_u_, vector_w_));

    norm_vector_u_ = vector_u_;
    norm_vector_v_ = vector_v_;
    norm_vector_w_ = normalize(vector_w_);

    const auto v_length = w_length * tanf(0.5f * state_.field_of_view * pi<float>() / 180.f);
    vector_v_ *= v_length;
    const auto u_length = v_length * state_.field_of_view;
    vector_u_ *= u_length;
    changed_ = false;
    return true;
}

void Camera::orbit(const float dx, const float dy) {
    vec3 t = state_.look_at - state_.eye;
    if (fabs(dot(normalize(t), vec3(0, 1, 0))) < 0.999f || t.y * dy < 0) {
        t = rotate(mat4(1.0f), dx * state_.look_speed * Timer::deltaTime, norm_vector_v_)
            * rotate(mat4(1.0f), dy * state_.look_speed * Timer::deltaTime, norm_vector_u_) * vec4(t, 1);
        state_.look_at = state_.eye + t;
        changed_ = true;
    }
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
        const auto mat = translate(mat4(1.0f), state_.move_speed * Timer::deltaTime * dir);
        state_.eye = mat * vec4(state_.eye, 1.0f);
        state_.look_at = mat * vec4(state_.look_at, 1.0f);
        changed_ = true;
    }
}
