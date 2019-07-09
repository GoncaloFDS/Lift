#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "per_ray_data.cuh"

// Note, the nomenclature used in the device code of all optixIntroduction samples
// follows some simple rules using prefixes to help indicating the scope and meaning:
//
// "sys" = renderer "system"-wide variables, defined at global context scope.
// "the" = variables with OptiX built-in semantic, like rtLaunchIndex, etc.
// "var" = "varyings" with developer defined attribute semantic, calculated by the intersection program.
// "par" = "parameter" variable held at some object scope, not at the global context scope.
//         (Exception to the last rule are the vertex "attributes" and "indices" held at Geometry nodes.)

rtBuffer<float4, 2> sysOutputBuffer; // RGBA32F

rtDeclareVariable(rtObject, sysTopObject, , );

rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

rtDeclareVariable(float3, sysCameraPosition, , );
rtDeclareVariable(float3, sysCameraU, , );
rtDeclareVariable(float3, sysCameraV, , );
rtDeclareVariable(float3, sysCameraW, , );

// Entry point for simple color filling kernel.
RT_PROGRAM void ray_generation() {
	PerRayData per_ray_data;
	per_ray_data.radiance = make_float3(0.0f);

	const float2 pixel = make_float2(theLaunchIndex);
	const float2 fragment = pixel + make_float2(0.5f);
	const float2 screen = make_float2(theLaunchDim);
	const float2 normalized_device_coords = (fragment / screen) * 2.0f - 1.0f;

	const float3 origin = sysCameraPosition;
	const float3 direction = optix::normalize(normalized_device_coords.x * sysCameraU + normalized_device_coords.y * sysCameraV + sysCameraW);

	optix::Ray ray = optix::make_Ray(origin, direction, 0, 0.0f, RT_DEFAULT_MAX);

	rtTrace(sysTopObject, ray, per_ray_data);
	sysOutputBuffer[theLaunchIndex] = make_float4(per_ray_data.radiance, 1.0f);
}
