#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "per_ray_data.cuh"

rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
rtDeclareVariable(PerRayData, thePerRayData, rtPayload, );

rtDeclareVariable(float3, sysColorBottom, , );
rtDeclareVariable(float3, sysColorTop, , );

RT_PROGRAM void miss_gradient() {
	const float t = theRay.direction.y * 0.5f + 0.5f;
	thePerRayData.radiance = optix::lerp(sysColorBottom, sysColorTop, t);
}