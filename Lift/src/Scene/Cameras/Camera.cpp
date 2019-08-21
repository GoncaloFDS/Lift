#include "pch.h"
#include "Camera.h"

Camera::Camera(): eye_(-10.0f, 2.0f, -12.f), look_at_(1.0f, 0.0f, 0.0f), up_(0.0f, 1.0f, 0.0f), fovy_(36.0f),
				  aspect_ratio_(1.0f) {
}

Camera::Camera(const vec3& eye, const vec3& look_at, const vec3& up, const float fovy, const float aspect_ratio):
	eye_(eye), look_at_(look_at), up_(up), fovy_(fovy), aspect_ratio_(aspect_ratio) {
}

void Camera::OnUpdate() {
	if (changed_) {
		vector_w_ = look_at_ - eye_; // Do not normalize -- it implies focal length
		const auto w_length = length(vector_w_);
		vector_u_ = normalize(cross(vector_w_, up_));
		vector_v_ = normalize(cross(vector_u_, vector_w_));

		const auto v_length = w_length * tanf(0.5f * fovy_ * pi<float>() / 180.f);
		vector_v_ *= v_length;
		const auto u_length = v_length * aspect_ratio_;
		vector_u_ *= u_length;
	}
}

void Camera::Orbit(const float dx, const float dy) {
	const auto t = look_at_ - eye_;
	if (fabs(dot(normalize(t), vec3(0, 1, 0))) < 0.999f || t.y * dy < 0) {
		eye_ = rotate(mat4(1.0f), dx * mouse_orbit_speed_, vector_v_)
			* rotate(mat4(1.0f), dy * mouse_orbit_speed_, vector_u_)
			* vec4(eye_, 1);
		changed_ = true;
	}
}

void Camera::Strafe(const float dx, const float dy) {
	const auto mat = translate(mat4(1.0f), dx * mouse_strafe_speed_ * vector_u_) *
		translate(mat4(1.0f), dy * mouse_strafe_speed_ * vector_v_);
	eye_ = mat * vec4(eye_, 1.0f);
	look_at_ = mat * vec4(look_at_, 1.0f);
	changed_ = true;
}

void Camera::Zoom(const float amount) {
	fovy_ += amount * mouse_zoom_speed_;
	fovy_ = std::clamp(fovy_, 0.001f, 180.0f);
	changed_ = true;
}
