#pragma once

namespace lift {

enum class Direction {
    UP, DOWN,
    RIGHT, LEFT,
    FORWARD, BACK
};

class Camera {
public:

    Camera();
    Camera(const vec3& eye, const vec3& look_at, const vec3& up, float fovy, float aspect_ratio);

    [[nodiscard]] auto direction() const -> vec3 { return normalize(look_at_ - eye_); }
    [[nodiscard]] auto eye() const -> const vec3& { return eye_; };
    [[nodiscard]] auto lookAt() const -> const vec3& { return look_at_; }
    [[nodiscard]] auto up() const -> const vec3& { return up_; }
    [[nodiscard]] auto fovy() const -> float { return fovy_; }
    [[nodiscard]] auto aspectRatio() const -> float { return aspect_ratio_; }

    void setDirection(const vec3& direction) { look_at_ = eye_ + length(look_at_ - eye_); }
    void setEye(const vec3& eye) { eye_ = eye; changed_ = true; }
    void setLookAt(const vec3& look_at) { look_at_ = look_at; changed_ = true; }
    void setUp(const vec3& up) { up_ = up; changed_ = true; }
    void setFovy(const float fovy) { fovy_ = fovy; changed_ = true; }
    void setAspectRatio(const float& aspect_ratio) { aspect_ratio_ = aspect_ratio; changed_ = true; }

    [[nodiscard]] auto vectorU() const -> const vec3& { return vector_u_; }
    [[nodiscard]] auto vectorV() const -> const vec3& { return vector_v_; }
    [[nodiscard]] auto vectorW() const -> const vec3& { return vector_w_; }
    auto onUpdate() -> bool;

    void orbit(float dx, float dy);
    void strafe(float dx, float dy);
    void zoom(float amount);
    void setMoveDirection(enum Direction direction, float amount = 1.0f);
private:

    vec3 eye_;
    vec3 look_at_;
    vec3 up_;
    float aspect_ratio_;
    float fovy_;

    vec3 vector_u_{1.0f};
    vec3 vector_v_{1.0f};
    vec3 vector_w_{1.0f};

    vec3 norm_vector_u_{1.0f};
    vec3 norm_vector_v_{1.0f};
    vec3 norm_vector_w_{1.0f};

    vec3 move_dir_{0.0f};

    float mouse_look_speed_{1.0f};
    float mouse_strafe_speed_{0.1f};
    float mouse_zoom_speed_{1.0f};
    float camera_move_speed_{3.0f};
    bool changed_ = true;

    void move();
};
}
