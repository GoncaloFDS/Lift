#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"

rtDeclareVariable(rtObject, sys_top_object, , );

rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );

rtDeclareVariable(RayPayload, the_ray_payload, rtPayload, );

//Attributes
rtDeclareVariable(optix::float3, var_geometry_normal, attribute GEO_NORMAL, );
//rtDeclareVariable(optix::float3, var_tangent, attribute TANGENT, );
rtDeclareVariable(optix::float3, var_normal, attribute NORMAL, );
//rtDeclareVariable(optix::float3, var_tex_coord, attribute TEXCOORD, );

RT_PROGRAM void closest_hit() {
	float3 geometry_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, var_geometry_normal));
	float3 normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, var_normal));

	if(0.0f < optix::dot(the_ray.direction, geometry_normal)) {
		normal = -normal;
	}
	//rtPrintf( "Hello from index !\n");
	the_ray_payload.radiance = normal * 0.5f + 0.5f;
}
