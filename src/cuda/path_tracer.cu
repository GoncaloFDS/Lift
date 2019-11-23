#include <optix.h>

#include "launch_parameters.h"
#include "vector_functions.hpp"
#include "random.cuh"
#include "vec_math.h"
#include "local_geometry.h"
#include "cuda_util.cuh"

using namespace lift;
extern "C" __constant__ LaunchParameters params;

static __forceinline__ __device__ PayloadRadiance* getPayload() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<PayloadRadiance*>( unpackPointer(u0, u1));
}

extern "C" __global__ void __raygen__render_frame() {
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const int frame_width = optixGetLaunchDimensions().x;
	const int frame_height = optixGetLaunchDimensions().y;
	const int accum_id = params.subframe_index;
	const auto& camera = params.camera;

	PayloadRadiance payload{};
	payload.pixel_color = make_float3(0.0f, 0.0f, 0.0f);
	payload.random_seed = tea<4>(ix + accum_id * frame_width, iy + accum_id * frame_height);

	uint32_t u0, u1;
	packPointer(&payload, u0, u1);

	float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
	for (int sample_id = 0; sample_id < params.samples_per_launch; sample_id++) {
		const float2 screen(make_float2(ix + rnd(payload.random_seed), iy + rnd(payload.random_seed))
								/ make_float2(frame_width, frame_height));

		float3 ray_dir = normalize(camera.w + (screen.x - 0.5f) * camera.u + (screen.y - 0.5f) * camera.v);

		optixTrace(params.handle,
				   camera.eye,
				   ray_dir,
				   0.0f,
				   1e20f,
				   0.0f,
				   OptixVisibilityMask(255),
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				   RADIANCE_RAY_TYPE,
				   RAY_TYPE_COUNT,
				   RADIANCE_RAY_TYPE,
				   u0, u1);
		pixel_color += payload.pixel_color;
	}
	const uint32_t image_index = ix + iy * frame_width;
	params.accum_buffer[image_index] = makeFloat4(pixel_color, 1.0f);
	params.frame_buffer[image_index] = makeColor(pixel_color);
}

extern "C" __global__ void __miss__radiance() {
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	PayloadRadiance* payload = getPayload();

	payload->pixel_color = makeFloat3(miss_data->bg_color);
}

extern "C" __global__ void __closesthit__occlusion() {
	//setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {
	const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
	const LocalGeometry geom = getLocalGeometry(hit_data->geometry_data);

	PayloadRadiance* payload = getPayload();

	float3 base_color = makeFloat3(hit_data->material_data.base_color);
	if (hit_data->material_data.base_color_tex) {
		base_color *=
			linearize(makeFloat3(tex2D<float4>(hit_data->material_data.base_color_tex, geom.uv.x, geom.uv.y)));
	}

	float metallic = hit_data->material_data.metallic;
	float roughness = hit_data->material_data.roughness;
	float4 mr_tex = makeFloat4(1.0f);
	if (hit_data->material_data.metallic_roughness_tex) {
		mr_tex = tex2D<float4>(hit_data->material_data.metallic_roughness_tex, geom.uv.x, geom.uv.y);
	}
	roughness *= mr_tex.y;
	metallic *= mr_tex.z;

	const float f_0 = 0.04f;
	const float3 diff_color = base_color * (1.0f - f_0) * (1.0f - metallic);
	const float3 spec_color = lerp(makeFloat3(f_0), base_color, metallic);
	const float alpha = roughness * roughness;

	float3 n = geom.n;
	if (hit_data->material_data.normal_tex) {
		const float4 nn = 2.0f * tex2D<float4>(hit_data->material_data.normal_tex, geom.uv.x, geom.uv.y)
			- makeFloat4(1.0f);
		n = normalize(nn.x * normalize(geom.dpdu) + nn.y * normalize(geom.dpdv) + nn.z * geom.n);
	}

	float3 pixel_color = makeFloat3(0.0f);

	const auto& light = params.lights[0];
	const int light_samples = 4;
	for (int light_sample = 0; light_sample < light_samples; light_sample++) {
		auto seed = payload->random_seed;
		const float3 light_pos = light.position
			+ rnd(seed) * light.v1
			+ rnd(seed) * light.v2;
		payload->random_seed = seed;

		float3 light_dir = light_pos - geom.p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		// shadow ray
		const float n_dot_l = dot(n, light_dir);
		if (n_dot_l >= 0.0f) {
			float3 light_visibility = makeFloat3(1.0f);
			uint32_t u0, u1;
			packPointer(&light_visibility, u0, u1);
			optixTrace(params.handle,
					   geom.p + 1e-3f * geom.ng,
					   light_dir,
					   1e-3f,
					   light_dist * (1.0f - 1e-3f),
					   0.0f,
					   OptixVisibilityMask(255),
					   OPTIX_RAY_FLAG_DISABLE_ANYHIT
						   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
						   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					   SHADOW_RAY_TYPE,
					   RAY_TYPE_COUNT,
					   SHADOW_RAY_TYPE,
					   u0, u1);
			pixel_color += light.intensity
				* light_visibility
				* diff_color
				* (n_dot_l / (light_dist * light_dist * light_samples));

		}
	}
	payload->pixel_color = pixel_color;

}















