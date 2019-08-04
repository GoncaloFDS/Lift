#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"
#include "light_definition.cuh"
#include "shader_common.cuh"

rtDeclareVariable(rtObject, sys_top_object, , );

rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );
rtDeclareVariable(float, the_intersection_distance, rtIntersectionDistance, );

rtDeclareVariable(RayPayload, the_payload, rtPayload, );

rtDeclareVariable(optix::float3, var_geometry_normal, attribute GEO_NORMAL, );
//rtDeclareVariable(optix::float3, var_tangent, attribute TANGENT, );
//rtDeclareVariable(optix::float3, var_normal, attribute NORMAL, );
//rtDeclareVariable(optix::float3, var_tex_coord, attribute TEXCOORD, );

rtBuffer<LightDefinition> sys_light_definitions;
rtDeclareVariable(int, per_light_index, , );

RT_PROGRAM void closest_hit_light() {
	the_payload.pos = the_ray.origin + the_ray.direction * the_intersection_distance;

	const optix::float3 geometry_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, var_geometry_normal));

	const float cos_theta = optix::dot(the_payload.wo, geometry_normal);
	the_payload.flags |= (cos_theta >= 0) ? FLAG_FRONTFACE : 0;

	the_payload.radiance = optix::make_float3(0.0f);

	if(the_payload.flags & FLAG_FRONTFACE) {
		const LightDefinition light = sys_light_definitions[per_light_index];

		the_payload.radiance = light.emission;
		const float pdf_light = (the_intersection_distance * the_intersection_distance) / (light.area * cos_theta);
		if((the_payload.flags & FLAG_DIFFUSE) && pdf_light > DENOMINATOR_EPSILON){
			the_payload.radiance *= powerHeuristic(the_payload.pdf, pdf_light);
		}
	}

	the_payload.flags |= FLAG_TERMINATE;
}

