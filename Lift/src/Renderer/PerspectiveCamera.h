#pragma once

enum class CameraState {
	None,
	Orbit,
	Dolly,
	Pan
};

class PerspectiveCamera {
public:
	PerspectiveCamera();

	void SetViewport(unsigned width, unsigned height);
	void SetBaseCoordinates(float x, float y);
	void SetSpeedRatio(float speed);
	void SetFocusDistance(float focus_distance);

	void Orbit(float x, float y);
	void Pan(float x, float y);
	void Dolly(float x, float y);
	void Focus(float x, float y);
	void Zoom(float x);

	void SetState(float x, float y, const CameraState& state);
	void SetState(const CameraState& state) { state_ = state; }
	CameraState& GetState() { return state_; }

	bool OnUpdate();
	[[nodiscard]] float GetAspectRatio() const;
	[[nodiscard]] vec3 GetPosition() const { return camera_position_; }
	[[nodiscard]] vec3 GetVectorU() const { return camera_u_; }
	[[nodiscard]] vec3 GetVectorV() const { return camera_v_; }
	[[nodiscard]] vec3 GetVectorW() const { return camera_w_; }

public: // Just to be able to load and save them easily.
	vec3 center_;
	// Center of interest point, around which is orbited (and the sharp plane of a depth of field camera).
	float focus_distance_; // Distance of the camera from the center of interest.
	float phi_; // Range [0.0f, 1.0f] from positive x-axis 360 degrees around the latitudes.
	float theta_; // Range [0.0f, 1.0f] from negative to positive y-axis.
	float fov_; // In degrees. Default is 60.0f

private:
	bool SetDelta(float x, float y);

private:
	unsigned width_; // Viewport width.
	unsigned height_; // Viewport height.
	float aspect_; // width / height
	float base_x_;
	float base_y_;
	float speed_ratio_;
	CameraState state_;

	// Derived values:
	float dx_;
	float dy_;
	bool changed_;
	vec3 camera_position_;
	vec3 camera_u_;
	vec3 camera_v_;
	vec3 camera_w_;
};
