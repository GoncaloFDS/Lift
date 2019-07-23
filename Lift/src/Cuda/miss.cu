#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"

rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );
rtDeclareVariable(RayPayload, the_ray_payload, rtPayload, );

rtDeclareVariable(float3, sys_color_top, , );
rtDeclareVariable(float3, sys_color_bottom, , );

RT_PROGRAM void miss_gradient() {
	const float t = the_ray.direction.y * 0.5f + 0.5f;
	the_ray_payload.radiance = optix::lerp(sys_color_bottom, sys_color_top, t);
}