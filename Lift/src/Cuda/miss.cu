#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"
#include "light_definition.cuh"
#include "shader_common.cuh"

rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );
rtDeclareVariable(RayPayload, the_payload, rtPayload, );

rtBuffer<LightDefinition> sys_light_definitions;

RT_PROGRAM void miss_environment_null() {
	the_payload.radiance = optix::make_float3(0.0f);
	the_payload.flags |= FLAG_TERMINATE;
}

RT_PROGRAM void miss_environment_constant() {
	const float mis_weight = (the_payload.flags & FLAG_DIFFUSE) ? powerHeuristic(the_payload.pdf, 0.25f * M_1_PIf) : 1.0f;
	the_payload.radiance = optix::make_float3(mis_weight);
	the_payload.flags |= FLAG_TERMINATE;
}