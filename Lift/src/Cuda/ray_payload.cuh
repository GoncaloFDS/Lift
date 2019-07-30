#pragma once

// Set if (0.0f <= wo_dot_ng), means looking onto the front face. (Edge-on is explicitly handled as frontface for the material stack.)
#define FLAG_FRONTFACE      0x00000010

// Highest bit set means terminate path.
#define FLAG_TERMINATE      0x80000000

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
