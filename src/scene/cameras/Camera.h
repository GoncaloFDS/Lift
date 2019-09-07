#pragma once

namespace lift {

enum class Direction {
    Up, Down,
    Right, Left,
    Forward, Back
};

class Camera {
public:

    Camera();
    Camera(const vec3& eye, const vec3& look_at, const vec3& up, const float fovy, const float aspect_ratio);

    void SetDirection(const vec3& direction) { look_at_ = eye_ + length(look_at_ - eye_); }
    [[nodiscard]] vec3 Direction() const { return normalize(look_at_ - eye_); }

    [[nodiscard]] const vec3& Eye() const { return eye_; };
    void SetEye(const vec3& eye) { eye_ = eye; }
    void SetLookAt(const vec3& look_at) { look_at_ = look_at; }
    [[nodiscard]] const vec3& LookAt() const { return look_at_; }
    [[nodiscard]] const vec3& Up() const { return up_; }
    void SetUp(const vec3& up) { up_ = up; }
    void SetFovy(const float fovy) { fovy_ = fovy; }
    [[nodiscard]] float Fovy() const { return fovy_; }
    [[nodiscard]] float AspectRatio() const { return aspect_ratio_; }
    void SetAspectRatio(const float& aspect_ratio) { aspect_ratio_ = aspect_ratio; }

    [[nodiscard]] const vec3& VectorU() const { return vector_u_; }

    [[nodiscard]] const vec3& VectorV() const { return vector_v_; }
    [[nodiscard]] const vec3& VectorW() const { return vector_w_; }
    void OnUpdate();

    void Orbit(float dx, float dy);
    void Strafe(float dx, float dy);
    void Zoom(float amount);
    void SetMoveDirection(enum Direction direction, float amount = 1.0f);
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

    float mouse_orbit_speed_{0.01f};
    float mouse_strafe_speed_{0.001f};
    float mouse_zoom_speed_{0.1f};
    float camera_move_speed_{0.1f};
    bool changed_ = true;

    void Move();
};
}
