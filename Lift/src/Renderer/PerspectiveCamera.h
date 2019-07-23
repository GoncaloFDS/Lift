#pragma once

class PerspectiveCamera {
public:
	PerspectiveCamera();

	void SetViewport(int width, int height);
	void SetBaseCoordinates(int x, int y);
	void SetSpeedRatio(float speed);
	void SetFocusDistance(float focus_distance);

	void Orbit(int x, int y);
	void Pan(int x, int y);
	void Dolly(int x, int y);
	void Focus(int x, int y);
	void Zoom(float x);

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
	bool SetDelta(int x, int y);

private:
	int width_; // Viewport width.
	int height_; // Viewport height.
	float aspect_; // width / height
	int base_x_;
	int base_y_;
	float speed_ratio_;

	// Derived values:
	int dx_;
	int dy_;
	bool changed_;
	vec3 camera_position_;
	vec3 camera_u_;
	vec3 camera_v_;
	vec3 camera_w_;
};
