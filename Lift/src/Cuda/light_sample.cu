#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "light_definition.cuh"
#include "shader_common.cuh"

rtBuffer<LightDefinition> sys_light_definitions;
rtDeclareVariable(int, sys_num_lights, , );

RT_FUNCTION void unitSquareToSphere(const float u, const float v, optix::float3& p, float& pdf) {
	p.z = 1.0f - 2.0f * u;
	float r = 1.0f - p.z * p.z;
	r = (0.0f < r) ? sqrtf(r) : 0.0f;

	const float phi = v * 2.0f * M_PIf;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	pdf = 0.25f * M_1_PIf; // == 1.0f / (4.0f * M_PIf)
}

// Note that all light sampling routines return lightSample.direction and lightSample.distance in world space!
RT_CALLABLE_PROGRAM void sample_light_constant(optix::float3 const& point, const optix::float2 sample, LightSample& light_sample) {
	unitSquareToSphere(sample.x, sample.y, light_sample.direction, light_sample.pdf);

	// Environment lights do not set the light sample position!
	light_sample.distance = RT_DEFAULT_MAX; // Environment light.

	// Explicit light sample. White scaled by inverse probabilty to hit this light.
	const LightDefinition light = sys_light_definitions[light_sample.index]; // The light index is picked by the caller!
	light_sample.emission = light.emission * float(sys_num_lights);
}


RT_CALLABLE_PROGRAM void sample_light_parallelogram(optix::float3 const& point, const optix::float2 sample, LightSample& light_sample) {
	light_sample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

	const LightDefinition light = sys_light_definitions[light_sample.index]; // The light index is picked by the caller!

	light_sample.position = light.position + light.vector_u * sample.x + light.vector_v * sample.y;
	// The light sample position in world coordinates.
	light_sample.direction = light_sample.position - point;
	// Sample direction from surface point to light sample position.
	light_sample.distance = optix::length(light_sample.direction);

	if ( light_sample.distance > DENOMINATOR_EPSILON ) {
		light_sample.direction /= light_sample.distance; // Normalized direction to light.

		const float cos_theta = optix::dot(-light_sample.direction, light.normal);
		if ( cos_theta > DENOMINATOR_EPSILON) { // Only emit light on the front side. 
			// Explicit light sample, must scale the emission by inverse probabilty to hit this light.
			light_sample.emission = light.emission * float(sys_num_lights);
			light_sample.pdf = (light_sample.distance * light_sample.distance) / (light.area * cos_theta);
			// Solid angle pdf. Assumes light.area != 0.0f.
		}
	}
}
