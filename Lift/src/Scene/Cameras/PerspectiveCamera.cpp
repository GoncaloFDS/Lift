#include "pch.h"
#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera()
	: center_(0.0f),
	  focus_distance_(10.0f),
	  phi_(0.75f), theta_(0.6f),
	  fov_(60.0f),
	  width_(1), height_(1),
	  aspect_(1.0f), base_x_(0),
	  base_y_(0),
	  speed_ratio_(10.0f),
	  state_(CameraState::None),
	  dx_(0), dy_(0),
	  changed_(true),
	  position_(0.0f),
	  vector_u_(1.0f, 0.0f, 0.0f),
	  vector_v_(0.0f, 1.0f, 0.0f),
	  vector_w_(1.0f, 0.0f, -1.0f) {
}

void PerspectiveCamera::SetViewport(const unsigned width, const unsigned height) {
	if (width_ == width && height_ == height)
		return; // Never drop to zero viewport. This avoids lots of checks for zero in other routines
	width_ = width ? width : 1;
	height_ = height ? height : 1;
	aspect_ = float(width_) / float(height_);
	changed_ = true;
}

void PerspectiveCamera::SetBaseCoordinates(const float x, const float y) {
	base_x_ = x;
	base_y_ = y;
}

void PerspectiveCamera::SetSpeedRatio(const float speed) {
	speed_ratio_ = speed;
	if (speed_ratio_ < 0.01f)
		speed_ratio_ = 0.01f;
	else if (speed_ratio_ > 100.f)
		speed_ratio_ = 100.f;
}

void PerspectiveCamera::SetFocusDistance(const float focus_distance) {
	if (focus_distance_ == focus_distance || 0.001f >= focus_distance)
		return;
	focus_distance_ = focus_distance;
	center_ = position_ + focus_distance_ * vector_w_;
	changed_ = true;
}

void PerspectiveCamera::Orbit(const float x, const float y) {
	if (!SetDelta(x, y))
		return;

	phi_ -= float(dx_) / float(width_); // Inverted
	// Wrap phi
	if (phi_ < 0.0f)
		phi_ += 1.0f;
	else if (phi_ > 1.0f)
		phi_ -= 1.0f;

	theta_ = std::clamp(theta_ + float(dy_) / float(height_), 0.0f, 1.0f);
}

void PerspectiveCamera::Pan(const float x, const float y) {
	if (SetDelta(x, y))
		center_ = center_ - float(dx_) / speed_ratio_ * vector_u_ + float(dy_) / speed_ratio_ * vector_v_;
}

void PerspectiveCamera::Dolly(const float x, const float y) {
	if (SetDelta(x, y)) {
		focus_distance_ -= float(dy_) / speed_ratio_ * length(vector_w_);
		if (focus_distance_ < 0.001f)
			focus_distance_ = 0.001f;
	}
}

void PerspectiveCamera::Focus(const float x, const float y) {
	if (SetDelta(x, y)) {
		SetFocusDistance(focus_distance_ - float(dy_) / speed_ratio_ * length(vector_w_));
	}
}

void PerspectiveCamera::Zoom(const float x) {
	fov_ = std::clamp(fov_ + float(x), 1.0f, 179.0f);
	changed_ = true;
}

void PerspectiveCamera::SetState(float x, float y, const CameraState& state) {
	base_x_ = x;
	base_y_ = y;
	state_ = state;
}

bool PerspectiveCamera::OnUpdate() {
	const auto changed = changed_;
	if (changed) {
		// Recalculate the camera parameters.
		
		const auto cos_phi = cosf(phi_ * 2.0f * pi<float>());
		const auto sin_phi = sinf(phi_ * 2.0f * pi<float>());
		const auto cos_theta = cosf(theta_ * pi<float>());
		const auto sin_theta = sinf(theta_ * pi<float>());

		const vec3 normal(cos_phi * sin_theta, -cos_theta, -sin_phi * sin_theta);
		// "normal", unit vector from origin to spherical coordinates (phi, theta)

		const auto tan_fov = tanf((fov_ * 0.5f) * pi<float>() / 180.0f); // m_fov is in the range [1.0f, 179.0f].
		position_ = center_ + focus_distance_ * normal;

		vector_u_ = aspect_ * vec3(-sin_phi, 0.0f, -cos_phi) * tan_fov; // "tangent"
		vector_v_ = vec3(cos_theta * cos_phi, sin_theta, cos_theta * -sin_phi) * tan_fov; // "bitangent"
		vector_w_ = -normal; // "-normal" to look at the center.

		changed_ = false; // Next time asking for the frustum will return false unless the camera has changed again.
	}
	return changed;
}

float PerspectiveCamera::GetAspectRatio() const {
	return aspect_;
}

bool PerspectiveCamera::SetDelta(const float x, const float y) {
	if (base_x_ != x || base_y_ != y) {
		dx_ = x - base_x_;
		dy_ = y - base_y_;

		base_x_ = x;
		base_y_ = y;

		changed_ = true;
		return true;
	}
	return false;
}
