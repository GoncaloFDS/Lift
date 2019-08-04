#pragma once

// Set by BSDFs which support direct lighting. Not set means specular interaction. Cleared in the closesthit program.
// Used to decide when to do direct lighting and multuiple importance sampling on implicit light hits.
#define FLAG_DIFFUSE        0x00000002

// Set if (0.0f <= wo_dot_ng), means looking onto the front face. (Edge-on is explicitly handled as frontface for the material stack.)
#define FLAG_FRONTFACE      0x00000010

// Highest bit set means terminate path.
#define FLAG_TERMINATE      0x80000000

// Keep flags active in a path segment which need to be tracked along the path.
// In this case only the last surface interaction is kept.
// It's needed to track the last bounce's diffuse state in case a ray hits a light implicitly for multiple importance sampling.
// FLAG_DIFFUSE is reset in the closest_hit program. 
#define FLAG_CLEAR_MASK     FLAG_DIFFUSE

struct RayPayload {
	optix::float3 pos; // Current surface hit point, in world space

	optix::float3 wo; // Outgoing direction, to observer, in world space.
	optix::float3 wi; // Incoming direction, to light, in world space.

	optix::float3 radiance; // Radiance along the current path segment.
	int flags; // Bitfield with flags. See FLAG_* defines for its contents.

	optix::float3 f_over_pdf; // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf; 
	float pdf; // The last BSDF sample's pdf, tracked for multiple importance sampling.

	unsigned int seed; // Random number generator input.
};

struct State {
	optix::float3 geometric_normal;
	optix::float3 normal;
};

struct ShadowRayPayload {
	bool visible;
};
