#include <optix.h>

#include "launch_parameters.h"
#include "vector_functions.hpp"
#include "random.cuh"
#include "vec_math.h"
#include "local_geometry.h"
#include "cuda_util.cuh"

using namespace lift;
extern "C" __constant__ LaunchParameters params;

static __forceinline__ __device__ RadiancePRD* getPRD() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<RadiancePRD*>( unpackPointer(u0, u1));
}

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
	optixSetPayload_0(static_cast<uint32_t>( occluded ));
}

static __forceinline__ __device__ void traceRadiance(OptixTraversableHandle handle,
													 float3 ray_origin,
													 float3 ray_direction,
													 float t_min,
													 float t_max,
													 RadiancePRD* prd) {
	uint32_t u0, u1;
	packPointer(prd, u0, u1);
	optixTrace(handle,
			   ray_origin,
			   ray_direction,
			   t_min,
			   t_max,
			   0.0f,
			   OptixVisibilityMask(1),
			   OPTIX_RAY_FLAG_NONE,
			   RADIANCE_RAY_TYPE,
			   RAY_TYPE_COUNT,
			   RADIANCE_RAY_TYPE,
			   u0, u1);

}

static __forceinline__ __device__ bool traceOcclusion(OptixTraversableHandle handle,
													  float3 ray_origin,
													  float3 ray_direction,
													  float t_min,
													  float t_max) {
	uint32_t occluded = 0u;
	optixTrace(handle,
			   ray_origin,
			   ray_direction,
			   t_min,
			   t_max,
			   0.0f,
			   OptixVisibilityMask(1),
			   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			   SHADOW_RAY_TYPE,
			   RAY_TYPE_COUNT,
			   SHADOW_RAY_TYPE,
			   occluded);
	return occluded;
}

extern "C" __global__ void __raygen__render_frame() {
	const int width = params.width;
	const int height = params.height;
	const float3 eye = params.camera.eye;
	const float3 U = params.camera.u;
	const float3 V = params.camera.v;
	const float3 W = params.camera.w;
	const uint3 idx = optixGetLaunchIndex();
	const int subframe_index = params.subframe_index;

	RadiancePRD prd{};
	prd.random.init(idx.y * width + idx.x, subframe_index);

	float3 result = makeFloat3(0.0f);
	int i = params.samples_per_launch;
	do {
		const float2 subpixel_jitter = make_float2(prd.random() - 0.5f, prd.random() - 0.5f);

		const float2 d = 2.0f * make_float2(
			(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(width),
			(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(height)
		) - 1.0f;
		float3 ray_direction = normalize(d.x * U + d.y * V + W);
		float3 ray_origin = eye;

		prd.emitted = makeFloat3(0.f);
		prd.radiance = makeFloat3(0.f);
		prd.attenuation = makeFloat3(1.f);
		prd.count_emitted = true;
		prd.done = false;

		int depth = 0;
		for (;;) {
			traceRadiance(
				params.handle,
				ray_origin,
				ray_direction,
				0.01f,  // tmin       // TODO: smarter offset
				1e16f,  // tmax
				&prd);

			result += prd.emitted;
			result += prd.radiance * prd.attenuation;

			if (prd.done || depth >= params.max_depth)
				break;

			ray_origin = prd.origin;
			ray_direction = prd.direction;

			++depth;
		}
	} while (--i);

	const uint3 launch_index = optixGetLaunchIndex();
	const uint32_t image_index = launch_index.y * params.width + launch_index.x;
	float3 accum_color = result / static_cast<float>( params.samples_per_launch );

	if (subframe_index > 0) {
		const float a = 1.0f / static_cast<float>( subframe_index + 1 );
		const float3 accum_color_prev = makeFloat3(params.accum_buffer[image_index]);
		accum_color = lerp(accum_color_prev, accum_color, a);
	}
	params.accum_buffer[image_index] = makeFloat4(accum_color, 1.0f);
	params.frame_buffer[image_index] = makeColor(accum_color);
}

extern "C" __global__ void __miss__radiance() {
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	RadiancePRD* prd = getPRD();

	prd->radiance = makeFloat3(miss_data->bg_color);
	prd->done = true;
}

extern "C" __global__ void __closesthit__occlusion() {
	setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {
	const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
	const LocalGeometry geom = getLocalGeometry(hit_data->geometry_data);

	const float3 ray_dir = optixGetWorldRayDirection();
	const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

	RadiancePRD* prd = getPRD();

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

	float3 N = geom.n;
	if (hit_data->material_data.normal_tex) {
		const float4 nn = 2.0f * tex2D<float4>(hit_data->material_data.normal_tex, geom.uv.x, geom.uv.y)
			- makeFloat4(1.0f);
		N = normalize(nn.x * normalize(geom.dpdu) + nn.y * normalize(geom.dpdv) + nn.z * geom.n);
	}

	if (prd->count_emitted) {
		prd->emitted = hit_data->material_data.emission_color;
	} else {
		prd->emitted = makeFloat3(0.0f);
	}

	{
		const float z1 = prd->random();
		const float z2 = prd->random();

		float3 w_in;
		cosine_sample_hemisphere(z1, z2, w_in);
		Onb onb(N);
		onb.inverse_transform(w_in);
		prd->direction = w_in;
		prd->origin = P;
		prd->attenuation *= diff_color;
		prd->count_emitted = false;

	}

	const float z1 = prd->random();
	const float z2 = prd->random();

	const auto& light = params.lights[0];

	const float3 light_pos = light.position + light.v1 * z1 + light.v2 * z2;

	const float L_dist = length(light_pos - P);
	const float3 L = normalize(light_pos - P);
	const float N_dot_L = dot(N, L);
	const float Ln_dot_L = -dot(light.normal, L);

	float weight = 0.0f;
	if (N_dot_L > 0.0f && Ln_dot_L > 0.0f) {
		const bool occluded = traceOcclusion(params.handle,
											 P,
											 L,
											 0.01f,
											 L_dist - 0.01f);
		if (!occluded) {
			const float A = length(cross(light.v1, light.v2));
			weight = N_dot_L * Ln_dot_L * A / (M_PIf * L_dist * L_dist);
		}
	}

	prd->radiance += light.emission * weight * 0.3f;
}















