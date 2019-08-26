#pragma once
#include "pch.h"

namespace lift {
	class Aabb {
	public:

		/** Construct an invalid box */
		Aabb();

		/** Construct from min and max vectors */
		Aabb(const vec3& min, const vec3& max);

		/** Construct from three points (e.g. triangle) */
		Aabb(const vec3& v0, const vec3& v1, const vec3& v2);

		/** Exact equality */

		bool operator==(const Aabb& other) const;

		/** Array access */
		vec3& operator[](int i);

		/** Const array access */
		const vec3& operator[](int i) const;

		/** Set using two vectors */

		void set(const vec3& min, const vec3& max);

		/** Set using three points (e.g. triangle) */

		void set(const vec3& v0, const vec3& v1, const vec3& v2);

		/** Invalidate the box */

		void invalidate();

		/** Check if the box is valid */

		bool valid() const;

		/** Check if the point is in the box */

		bool contains(const vec3& p) const;

		/** Check if the box is fully contained in the box */

		bool contains(const Aabb& bb) const;

		/** Extend the box to include the given point */

		void include(const vec3& p);

		/** Extend the box to include the given box */

		void include(const Aabb& other);

		/** Extend the box to include the given box */

		void include(const vec3& min, const vec3& max);

		/** Compute the box center */
		vec3 center() const;

		/** Compute the box center in the given dimension */

		float center(int dim) const;

		/** Compute the box extent */
		vec3 extent() const;

		/** Compute the box extent in the given dimension */

		float extent(int dim) const;

		/** Compute the volume of the box */

		float volume() const;

		/** Compute the surface area of the box */

		float area() const;

		/** Compute half the surface area of the box */

		float halfArea() const;

		/** Get the index of the longest axis */

		int longestAxis() const;

		/** Get the extent of the longest axis */

		float maxExtent() const;

		/** Check for intersection with another box */

		bool intersects(const Aabb& other) const;

		/** Make the current box be the intersection between this one and another one */

		void intersection(const Aabb& other);

		/** Enlarge the box by moving both min and max by 'amount' */

		void enlarge(float amount);


		void transform(const mat4& m);

		/** Check if the box is flat in at least one dimension  */

		bool isFlat() const;

		/** Compute the minimum Euclidean distance from a point on the
		 surface of this Aabb to the point of interest */

		float distance(const vec3& x) const;

		/** Compute the minimum squared Euclidean distance from a point on the
		 surface of this Aabb to the point of interest */

		float distance2(const vec3& x) const;

		/** Compute the minimum Euclidean distance from a point on the surface
		  of this Aabb to the point of interest.
		  If the point of interest lies inside this Aabb, the result is negative  */

		float signedDistance(const vec3& x) const;

		/** Min bound */
		vec3 m_min;
		/** Max bound */
		vec3 m_max;
	};


	inline Aabb::Aabb() {
		invalidate();
	}

	inline Aabb::Aabb(const vec3& min, const vec3& max) {
		set(min, max);
	}

	inline Aabb::Aabb(const vec3& v0, const vec3& v1, const vec3& v2) {
		set(v0, v1, v2);
	}

	inline

	bool Aabb::operator==(const Aabb& other) const {
		return m_min.x == other.m_min.x &&
			m_min.y == other.m_min.y &&
			m_min.z == other.m_min.z &&
			m_max.x == other.m_max.x &&
			m_max.y == other.m_max.y &&
			m_max.z == other.m_max.z;
	}

	inline vec3& Aabb::operator[](int i) {
		LF_ASSERT(i >= 0 && i <= 1, "");
		return (&m_min)[i];
	}

	inline

		const vec3& Aabb::operator[](int i) const {
		LF_ASSERT(i >= 0 && i <= 1, "");
		return (&m_min)[i];
	}

	inline

	void Aabb::set(const vec3& min, const vec3& max) {
		m_min = min;
		m_max = max;
	}

	inline

	void Aabb::set(const vec3& v0, const vec3& v1, const vec3& v2) {
		m_min = min(v0, min(v1, v2));
		m_max = max(v0, max(v1, v2));
	}

	inline

	void Aabb::invalidate() {
		m_min = vec3(1e37f);
		m_max = vec3(-1e37f);
	}

	inline

	bool Aabb::valid() const {
		return m_min.x <= m_max.x &&
			m_min.y <= m_max.y &&
			m_min.z <= m_max.z;
	}

	inline

	bool Aabb::contains(const vec3& p) const {
		return p.x >= m_min.x && p.x <= m_max.x &&
			p.y >= m_min.y && p.y <= m_max.y &&
			p.z >= m_min.z && p.z <= m_max.z;
	}

	inline

	bool Aabb::contains(const Aabb& bb) const {
		return contains(bb.m_min) && contains(bb.m_max);
	}

	inline

	void Aabb::include(const vec3& p) {
		m_min = min(m_min, p);
		m_max = max(m_max, p);
	}

	inline

	void Aabb::include(const Aabb& other) {
		m_min = min(m_min, other.m_min);
		m_max = max(m_max, other.m_max);
	}

	inline

	void Aabb::include(const vec3& min, const vec3& max) {
		m_min = glm::min(m_min, min);
		m_max = glm::max(m_max, max);
	}

	inline vec3 Aabb::center() const {
		LF_ASSERT(valid(), "");
		return (m_min + m_max) * 0.5f;
	}

	inline

	float Aabb::center(int dim) const {
		LF_ASSERT(valid(), "");
		LF_ASSERT(dim >= 0 && dim <= 2, "");
		return (((const float*)(&m_min))[dim] + ((const float*)(&m_max))[dim]) * 0.5f;
	}

	inline vec3 Aabb::extent() const {
		LF_ASSERT(valid(), "");
		return m_max - m_min;
	}

	inline

	float Aabb::extent(int dim) const {
		LF_ASSERT(valid(), "");
		return ((const float*)(&m_max))[dim] - ((const float*)(&m_min))[dim];
	}

	inline

	float Aabb::volume() const {
		LF_ASSERT(valid(), "");
		const vec3 d = extent();
		return d.x * d.y * d.z;
	}

	inline

	float Aabb::area() const {
		return 2.0f * halfArea();
	}

	inline

	float Aabb::halfArea() const {
		LF_ASSERT(valid(), "");
		const vec3 d = extent();
		return d.x * d.y + d.y * d.z + d.z * d.x;
	}

	inline

	int Aabb::longestAxis() const {
		LF_ASSERT(valid(), "");
		const vec3 d = extent();

		if (d.x > d.y)
			return d.x > d.z ? 0 : 2;
		return d.y > d.z ? 1 : 2;
	}

	inline

	float Aabb::maxExtent() const {
		return extent(longestAxis());
	}

	inline

	bool Aabb::intersects(const Aabb& other) const {
		if (other.m_min.x > m_max.x || other.m_max.x < m_min.x) return false;
		if (other.m_min.y > m_max.y || other.m_max.y < m_min.y) return false;
		if (other.m_min.z > m_max.z || other.m_max.z < m_min.z) return false;
		return true;
	}

	inline

	void Aabb::intersection(const Aabb& other) {
		m_min.x = max(m_min.x, other.m_min.x);
		m_min.y = max(m_min.y, other.m_min.y);
		m_min.z = max(m_min.z, other.m_min.z);
		m_max.x = min(m_max.x, other.m_max.x);
		m_max.y = min(m_max.y, other.m_max.y);
		m_max.z = min(m_max.z, other.m_max.z);
	}

	inline

	void Aabb::enlarge(float amount) {
		LF_ASSERT(valid(), "");
		m_min -= vec3(amount);
		m_max += vec3(amount);
	}

	inline

	void Aabb::transform(const mat4& m) {
		const vec3 b000 = m_min;
		const vec3 b001 = vec3(m_min.x, m_min.y, m_max.z);
		const vec3 b010 = vec3(m_min.x, m_max.y, m_min.z);
		const vec3 b011 = vec3(m_min.x, m_max.y, m_max.z);
		const vec3 b100 = vec3(m_max.x, m_min.y, m_min.z);
		const vec3 b101 = vec3(m_max.x, m_min.y, m_max.z);
		const vec3 b110 = vec3(m_max.x, m_max.y, m_min.z);
		const vec3 b111 = m_max;

		invalidate();
		include(vec3(m * vec4(b000, 1.0f)));
		include(vec3(m * vec4(b001, 1.0f)));
		include(vec3(m * vec4(b010, 1.0f)));
		include(vec3(m * vec4(b011, 1.0f)));
		include(vec3(m * vec4(b100, 1.0f)));
		include(vec3(m * vec4(b101, 1.0f)));
		include(vec3(m * vec4(b110, 1.0f)));
		include(vec3(m * vec4(b111, 1.0f)));
	}

	inline

	bool Aabb::isFlat() const {
		return m_min.x == m_max.x ||
			m_min.y == m_max.y ||
			m_min.z == m_max.z;
	}

	inline

	float Aabb::distance(const vec3& x) const {
		return sqrtf(distance2(x));
	}

	inline

	float Aabb::signedDistance(const vec3& x) const {
		if (m_min.x <= x.x && x.x <= m_max.x &&
			m_min.y <= x.y && x.y <= m_max.y &&
			m_min.z <= x.z && x.z <= m_max.z) {
			float distance_x = min(x.x - m_min.x, m_max.x - x.x);
			float distance_y = min(x.y - m_min.y, m_max.y - x.y);
			float distance_z = min(x.z - m_min.z, m_max.z - x.z);

			float min_distance = min(distance_x, min(distance_y, distance_z));
			return -min_distance;
		}

		return distance(x);
	}

	inline

	float Aabb::distance2(const vec3& x) const {
		vec3 box_dims = m_max - m_min;

		// compute vector from min corner of box
		vec3 v = x - m_min;

		float dist2 = 0;
		float excess;

		// project vector from box min to x on each axis,
		// yielding distance to x along that axis, and count
		// any excess distance outside box extents

		excess = 0;
		if (v.x < 0)
			excess = v.x;
		else if (v.x > box_dims.x)
			excess = v.x - box_dims.x;
		dist2 += excess * excess;

		excess = 0;
		if (v.y < 0)
			excess = v.y;
		else if (v.y > box_dims.y)
			excess = v.y - box_dims.y;
		dist2 += excess * excess;

		excess = 0;
		if (v.z < 0)
			excess = v.z;
		else if (v.z > box_dims.z)
			excess = v.z - box_dims.z;
		dist2 += excess * excess;

		return dist2;
	}

} // end namespace sutil
