#include <optixu/optixu_math_namespace.h>
#include "random.cuh"
#include "ParallelogramLight.cuh"

using namespace optix;

// Common optix::Ray types
#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#include <optixu/optixu_vector_types.h>

struct BasicLight {
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;
	float3 color;
	int casts_shadow;
	int padding; // make this structure 32 bytes -- powers of two are your friend!
};

struct PerRayData_pathtrace {
	float3 result;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
};

struct PerRayData_pathtrace_shadow {
	bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );

rtBuffer<float4, 2> output_buffer;
rtBuffer<ParallelogramLight> lights;


RT_PROGRAM void pathtrace_camera() {
	size_t2 screen = output_buffer.size();

	float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;
	unsigned int samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
	float3 result = make_float3(0.0f);

	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);
	do {
		//
		// Sample pixel using jittering
		//
		unsigned int x = samples_per_pixel % sqrt_num_samples;
		unsigned int y = samples_per_pixel / sqrt_num_samples;
		float2 jitter = make_float2(x - rnd(seed), y - rnd(seed));
		float2 d = pixel + jitter * jitter_scale;
		float3 ray_origin = eye;
		float3 ray_direction = normalize(d.x * U + d.y * V + W);

		// Initialze per-ray data
		PerRayData_pathtrace prd;
		prd.result = make_float3(0.f);
		prd.attenuation = make_float3(1.f);
		prd.countEmitted = true;
		prd.done = false;
		prd.seed = seed;
		prd.depth = 0;

		// Each iteration is a segment of the ray path.  The closest hit will
		// return new segments to be traced here.
		for (;;) {
			Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(top_object, ray, prd);

			if (prd.done) {
				// We have hit the background or a luminaire
				prd.result += prd.radiance * prd.attenuation;
				break;
			}

			// Russian roulette termination 
			if (prd.depth >= rr_begin_depth) {
				float pcont = fmaxf(prd.attenuation);
				if (rnd(prd.seed) >= pcont)
					break;
				prd.attenuation /= pcont;
			}

			prd.depth++;
			prd.result += prd.radiance * prd.attenuation;

			// Update ray data for the next path segment
			ray_origin = prd.origin;
			ray_direction = prd.direction;
		}

		result += prd.result;
		seed = prd.seed;
	}
	while (--samples_per_pixel);

	//
	// Update the output buffer
	//
	float3 pixel_color = result / (sqrt_num_samples * sqrt_num_samples);

	if (frame_number > 1) {
		float a = 1.0f / (float)frame_number;
		float3 old_color = make_float3(output_buffer[launch_index]);
		output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
	}
	else {
		output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
	}
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, emission_color, , );

RT_PROGRAM void diffuseEmitter() {
	current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
	current_prd.done = true;
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );


RT_PROGRAM void diffuse() {
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;

	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	current_prd.attenuation = current_prd.attenuation * diffuse_color;
	current_prd.countEmitted = false;

	//
	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	for (int i = 0; i < num_lights; ++i) {
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

		// Calculate properties of light sample (for area based pdf)
		const float Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float nDl = dot(ffnormal, L);
		const float LnDl = dot(light.normal, L);

		// cast shadow ray
		if (nDl > 0.0f && LnDl > 0.0f) {
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_object, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow) {
				const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
				result += light.emission * weight;
			}
		}
	}

	current_prd.radiance = result;
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow() {
	current_prd_shadow.inShadow = true;
	rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception() {
	output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss() {
	current_prd.radiance = bg_color;
	current_prd.done = true;
}
