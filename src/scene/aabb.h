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
    Aabb(const vec3& v_0, const vec3& v_1, const vec3& v_2);

    /** Exact equality */

    auto operator==(const Aabb& other) const -> bool;

    /** Array access */
    auto operator[](int i) -> vec3&;

    /** Const array access */
    auto operator[](int i) const -> const vec3&;

    /** Set using two vectors */
    void set(const vec3& min, const vec3& max);

    /** Set using three points (e.g. triangle) */
    void set(const vec3& v_0, const vec3& v_1, const vec3& v_2);

    /** Invalidate the box */
    void invalidate();

    /** Check if the box is valid */
    [[nodiscard]] auto valid() const -> bool;

    /** Check if the point is in the box */
    [[nodiscard]] auto contains(const vec3& p) const -> bool;

    /** Check if the box is fully contained in the box */
    [[nodiscard]] auto contains(const Aabb& bb) const -> bool;

    /** Extend the box to include the given point */
    void include(const vec3& p);

    /** Extend the box to include the given box */
    void include(const Aabb& other);

    /** Extend the box to include the given box */
    void include(const vec3& min, const vec3& max);

    /** Compute the box center */
    [[nodiscard]] auto center() const -> vec3;

    /** Compute the box center in the given dimension */
    [[nodiscard]] auto center(int dim) const -> float;

    /** Compute the box extent */
    [[nodiscard]] auto extent() const -> vec3;

    /** Compute the box extent in the given dimension */
    [[nodiscard]] auto extent(int dim) const -> float;

    /** Compute the volume of the box */
    [[nodiscard]] auto volume() const -> float;

    /** Compute the surface area of the box */
    [[nodiscard]] auto area() const -> float;

    /** Compute half the surface area of the box */
    [[nodiscard]] auto halfArea() const -> float;

    /** Get the index of the longest axis */
    [[nodiscard]] auto longestAxis() const -> int;

    /** Get the extent of the longest axis */
    [[nodiscard]] auto maxExtent() const -> float;

    /** Check for intersection with another box */
    [[nodiscard]] auto intersects(const Aabb& other) const -> bool;

    /** Make the current box be the intersection between this one and another one */
    void intersection(const Aabb& other);

    /** Enlarge the box by moving both min and max by 'amount' */
    void enlarge(float amount);

    void transform(const mat4& m);

    /** Check if the box is flat in at least one dimension  */
    [[nodiscard]] auto isFlat() const -> bool;

    /** Compute the minimum Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    [[nodiscard]] auto distance(const vec3& x) const -> float;

    /** Compute the minimum squared Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    [[nodiscard]] auto distance2(const vec3& x) const -> float;

    /** Compute the minimum Euclidean distance from a point on the surface
      of this Aabb to the point of interest.
      If the point of interest lies inside this Aabb, the result is negative  */
    [[nodiscard]] auto signedDistance(const vec3& x) const -> float;

    /** Min bound */
    vec3 min_{};
    /** Max bound */
    vec3 max_{};
};

}

inline lift::Aabb::Aabb() {
    invalidate();
}

inline lift::Aabb::Aabb(const vec3& min, const vec3& max) {
    set(min, max);
}

inline lift::Aabb::Aabb(const vec3& v_0, const vec3& v_1, const vec3& v_2) {
    set(v_0, v_1, v_2);
}

inline auto lift::Aabb::operator==(const Aabb& other) const -> bool {
    return min_.x == other.min_.x &&
        min_.y == other.min_.y &&
        min_.z == other.min_.z &&
        max_.x == other.max_.x &&
        max_.y == other.max_.y &&
        max_.z == other.max_.z;
}

inline auto lift::Aabb::operator[](int i) -> vec3& {
    LF_ASSERT(i >= 0 && i <= 1, "");
    return (&min_)[i];
}

inline auto lift::Aabb::operator[](int i) const -> const vec3& {
    LF_ASSERT(i >= 0 && i <= 1, "");
    return (&min_)[i];
}

inline void lift::Aabb::set(const vec3& min, const vec3& max) {
    min_ = min;
    max_ = max;
}

inline void lift::Aabb::set(const vec3& v_0, const vec3& v_1, const vec3& v_2) {
    min_ = min(v_0, min(v_1, v_2));
    max_ = max(v_0, max(v_1, v_2));
}

inline void lift::Aabb::invalidate() {
    min_ = vec3(1e37f);
    max_ = vec3(-1e37f);
}

inline auto lift::Aabb::valid() const -> bool {
    return min_.x <= max_.x &&
        min_.y <= max_.y &&
        min_.z <= max_.z;
}

inline auto lift::Aabb::contains(const vec3& p) const -> bool {
    return p.x >= min_.x && p.x <= max_.x &&
        p.y >= min_.y && p.y <= max_.y &&
        p.z >= min_.z && p.z <= max_.z;
}

inline auto lift::Aabb::contains(const Aabb& bb) const -> bool {
    return contains(bb.min_) && contains(bb.max_);
}

inline void lift::Aabb::include(const vec3& p) {
    min_ = min(min_, p);
    max_ = max(max_, p);
}

inline void lift::Aabb::include(const Aabb& other) {
    min_ = min(min_, other.min_);
    max_ = max(max_, other.max_);
}

inline void lift::Aabb::include(const vec3& min, const vec3& max) {
    min_ = glm::min(min_, min);
    max_ = glm::max(max_, max);
}

inline auto lift::Aabb::center() const -> vec3 {
    LF_ASSERT(valid(), "");
    return (min_ + max_) * 0.5f;
}

inline auto lift::Aabb::center(int dim) const -> float {
    LF_ASSERT(valid(), "");
    LF_ASSERT(dim >= 0 && dim <= 2, "");
    return (((const float*) (&min_))[dim] + ((const float*) (&max_))[dim]) * 0.5f;
}

inline auto lift::Aabb::extent() const -> vec3 {
    LF_ASSERT(valid(), "");
    return max_ - min_;
}

inline auto lift::Aabb::extent(int dim) const -> float {
    LF_ASSERT(valid(), "");
    return ((const float*) (&max_))[dim] - ((const float*) (&min_))[dim];
}

inline auto lift::Aabb::volume() const -> float {
    LF_ASSERT(valid(), "");
    const vec3 d = extent();
    return d.x * d.y * d.z;
}

inline auto lift::Aabb::area() const -> float {
    return 2.0f * halfArea();
}

inline auto lift::Aabb::halfArea() const -> float {
    LF_ASSERT(valid(), "");
    const vec3 d = extent();
    return d.x * d.y + d.y * d.z + d.z * d.x;
}

inline auto lift::Aabb::longestAxis() const -> int {
    LF_ASSERT(valid(), "");
    const vec3 d = extent();

    if (d.x > d.y)
        return d.x > d.z ? 0 : 2;
    return d.y > d.z ? 1 : 2;
}

inline auto lift::Aabb::maxExtent() const -> float {
    return extent(longestAxis());
}

inline auto lift::Aabb::intersects(const Aabb& other) const -> bool {
    if (other.min_.x > max_.x || other.max_.x < min_.x) return false;
    if (other.min_.y > max_.y || other.max_.y < min_.y) return false;
    return !(other.min_.z > max_.z || other.max_.z < min_.z);
}

inline void lift::Aabb::intersection(const Aabb& other) {
    min_.x = max(min_.x, other.min_.x);
    min_.y = max(min_.y, other.min_.y);
    min_.z = max(min_.z, other.min_.z);
    max_.x = min(max_.x, other.max_.x);
    max_.y = min(max_.y, other.max_.y);
    max_.z = min(max_.z, other.max_.z);
}

inline void lift::Aabb::enlarge(float amount) {
    LF_ASSERT(valid(), "");
    min_ -= vec3(amount);
    max_ += vec3(amount);
}

inline void lift::Aabb::transform(const mat4& m) {
    const vec3 b_000 = min_;
    const vec3 b_001 = vec3(min_.x, min_.y, max_.z);
    const vec3 b_010 = vec3(min_.x, max_.y, min_.z);
    const vec3 b_011 = vec3(min_.x, max_.y, max_.z);
    const vec3 b_100 = vec3(max_.x, min_.y, min_.z);
    const vec3 b_101 = vec3(max_.x, min_.y, max_.z);
    const vec3 b_110 = vec3(max_.x, max_.y, min_.z);
    const vec3 b_111 = max_;

    invalidate();
    include(vec3(m * vec4(b_000, 1.0f)));
    include(vec3(m * vec4(b_001, 1.0f)));
    include(vec3(m * vec4(b_010, 1.0f)));
    include(vec3(m * vec4(b_011, 1.0f)));
    include(vec3(m * vec4(b_100, 1.0f)));
    include(vec3(m * vec4(b_101, 1.0f)));
    include(vec3(m * vec4(b_110, 1.0f)));
    include(vec3(m * vec4(b_111, 1.0f)));
}

inline bool lift::Aabb::isFlat() const {
    return min_.x == max_.x ||
        min_.y == max_.y ||
        min_.z == max_.z;
}

inline auto lift::Aabb::distance(const vec3& x) const -> float {
    return sqrtf(distance2(x));
}

inline auto lift::Aabb::signedDistance(const vec3& x) const -> float {
    if (min_.x <= x.x && x.x <= max_.x &&
        min_.y <= x.y && x.y <= max_.y &&
        min_.z <= x.z && x.z <= max_.z) {
        float distance_x = min(x.x - min_.x, max_.x - x.x);
        float distance_y = min(x.y - min_.y, max_.y - x.y);
        float distance_z = min(x.z - min_.z, max_.z - x.z);

        float min_distance = min(distance_x, min(distance_y, distance_z));
        return -min_distance;
    }

    return distance(x);
}

inline auto lift::Aabb::distance2(const vec3& x) const -> float {
    vec3 box_dims = max_ - min_;

    // compute vector from min corner of box
    vec3 v = x - min_;

    float dist_2 = 0;
    float excess;

    // project vector from box min to x on each axis,
    // yielding distance to x along that axis, and count
    // any excess distance outside box extents

    excess = 0;
    if (v.x < 0)
        excess = v.x;
    else if (v.x > box_dims.x)
        excess = v.x - box_dims.x;
    dist_2 += excess * excess;

    excess = 0;
    if (v.y < 0)
        excess = v.y;
    else if (v.y > box_dims.y)
        excess = v.y - box_dims.y;
    dist_2 += excess * excess;

    excess = 0;
    if (v.z < 0)
        excess = v.z;
    else if (v.z > box_dims.z)
        excess = v.z - box_dims.z;
    dist_2 += excess * excess;

    return dist_2;
}
