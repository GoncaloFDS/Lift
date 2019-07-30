#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"
#include "random.cuh"
#include "shader_common.cuh"

// Note, the nomenclature used in the device code of all optixIntroduction samples
// follows some simple rules using prefixes to help indicating the scope and meaning:
//
// "sys" = renderer "system"-wide variables, defined at global context scope.
// "the" = variables with OptiX built-in semantic, like rtLaunchIndex, etc.
// "var" = "varyings" with developer defined attribute semantic, calculated by the intersection program.
// "par" = "parameter" variable held at some object scope, not at the global context scope.
//         (Exception to the last rule are the vertex "attributes" and "indices" held at Geometry nodes.)

rtBuffer<float4, 2> sys_output_buffer; // RGBA32F

rtDeclareVariable(rtObject, sys_top_object, , );

rtDeclareVariable(float, sys_scene_epsilon, , );
rtDeclareVariable(int2, sys_path_lengths, , );
rtDeclareVariable(int, sys_iteration_index, , );

rtDeclareVariable(uint2, the_launch_dimension, rtLaunchDim, );
rtDeclareVariable(uint2, the_launch_index, rtLaunchIndex, );

rtDeclareVariable(float3, sys_camera_position, , );
rtDeclareVariable(float3, sys_camera_u, , );
rtDeclareVariable(float3, sys_camera_v, , );
rtDeclareVariable(float3, sys_camera_w, , );

RT_FUNCTION void integrator(RayPayload& payload, float3& radiance) {
	radiance = make_float3(0.0f);
	float3 throughput = make_float3(1.0f);
	int depth = 0;

	while (depth < sys_path_lengths.y) {
		payload.wo = -payload.wi;
		payload.flags = 0;

		optix::Ray ray = optix::make_Ray(payload.pos, payload.wi, 0, sys_scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(sys_top_object, ray, payload);

		radiance += throughput * payload.radiance;

		if((payload.flags & FLAG_TERMINATE) || payload.pdf <= 0.0f || isNull(payload.f_over_pdf)) {
			break;
		}

		throughput *= payload.f_over_pdf;
		++depth;
	}
}

// Entry point for simple color filling kernel.
RT_PROGRAM void ray_generation() {
	RayPayload payload;
	payload.seed = tea<8>(the_launch_index.y * the_launch_dimension.x + the_launch_index.x, sys_iteration_index);
	payload.radiance = make_float3(0.0f);

	const float2 pixel = make_float2(the_launch_index);
	const float2 fragment = pixel + rng2(payload.seed);
	const float2 screen = make_float2(the_launch_dimension);
	const float2 normalized_device_coords = (fragment / screen) * 2.0f - 1.0f;

	payload.pos = sys_camera_position;
	payload.wi = optix::normalize(normalized_device_coords.x * sys_camera_u + normalized_device_coords.y * sys_camera_v + sys_camera_w);

	float3 radiance;

	integrator(payload, radiance);

	if(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)) {
		radiance = make_float3(1000000.0f, 0.0f, 0.0f);
	}
	else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
		radiance = make_float3(0.0f, 1000000.0f, 0.0f);
	}
	else if( radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f) {
		radiance = make_float3(0.0f, 0.0f, 1000000.0f);
	}

	if(!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))){
		if(0 < sys_iteration_index) {
			float4 dst = sys_output_buffer[the_launch_index];
			sys_output_buffer[the_launch_index] = optix::lerp(dst, make_float4(radiance, 1.0f), 1.0f/(float)(sys_iteration_index + 1));
		}
		else {
			sys_output_buffer[the_launch_index] = make_float4(radiance, 1.0f);
		}
	}
}
