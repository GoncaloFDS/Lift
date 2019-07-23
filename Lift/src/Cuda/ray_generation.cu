#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"

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

rtDeclareVariable(uint2, the_launch_dimension, rtLaunchDim, );
rtDeclareVariable(uint2, the_launch_index, rtLaunchIndex, );

rtDeclareVariable(float3, sys_camera_position, , );
rtDeclareVariable(float3, sys_camera_u, , );
rtDeclareVariable(float3, sys_camera_v, , );
rtDeclareVariable(float3, sys_camera_w, , );

// Entry point for simple color filling kernel.
RT_PROGRAM void ray_generation() {
	RayPayload per_ray_data;
	per_ray_data.radiance = make_float3(0.0f);

	const float2 pixel = make_float2(the_launch_index);
	const float2 fragment = pixel + make_float2(0.5f);
	const float2 screen = make_float2(the_launch_dimension);
	const float2 normalized_device_coords = (fragment / screen) * 2.0f - 1.0f;

	const float3 origin = sys_camera_position;
	const float3 direction = optix::normalize(normalized_device_coords.x * sys_camera_u + normalized_device_coords.y * sys_camera_v + sys_camera_w);

	optix::Ray ray = optix::make_Ray(origin, direction, 0, 0.0f, RT_DEFAULT_MAX);

	rtTrace(sys_top_object, ray, per_ray_data);
	sys_output_buffer[the_launch_index] = make_float4(per_ray_data.radiance, 1.0f);
}
