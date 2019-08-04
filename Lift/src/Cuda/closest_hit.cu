#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "ray_payload.cuh"
#include "random.cuh"
#include "material_parameter.cuh"
#include "light_definition.cuh"
#include "shader_common.cuh"

// Context global variables provided by the host
rtDeclareVariable(rtObject, sys_top_object, , );
rtDeclareVariable(float, sys_scene_epsilon, , );

// Semantic variables
rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );
rtDeclareVariable(float, the_intersection_distance, rtIntersectionDistance, );

rtDeclareVariable(RayPayload, the_payload, rtPayload, );

//Attributes
rtDeclareVariable(optix::float3, var_geometry_normal, attribute GEO_NORMAL, );
//rtDeclareVariable(optix::float3, var_tangent, attribute TANGENT, );
rtDeclareVariable(optix::float3, var_normal, attribute NORMAL, );
//rtDeclareVariable(optix::float3, var_tex_coord, attribute TEXCOORD, );

rtBuffer<MaterialParameter> sys_material_parameters;
rtDeclareVariable(int, per_material_index, , );

rtBuffer<LightDefinition> sys_light_definitions;
rtDeclareVariable(int, sys_num_lights, , );

rtBuffer<rtCallableProgramId<void(float3 const& point, const float2 sample, LightSample& light_sample)> > sys_sample_light;

RT_FUNCTION void align_vector(float3 const& axis, float3& w) {
	// Align w with axis.
	const float s = copysign(1.0f, axis.z);
	w.z *= s;
	const float3 h = make_float3(axis.x, axis.y, axis.z + s);
	const float k = optix::dot(w, h) / (1.0f + fabsf(axis.z));
	w = k * h - w;
}

RT_FUNCTION void unit_square_to_cosine_hemisphere(const float2 sample, float3 const& axis, float3& w, float& pdf) {
	// Choose a point on the local hemisphere coordinates about +z.
	const float theta = 2.0f * M_PIf * sample.x;
	const float r = sqrtf(sample.y);
	w.x = r * cosf(theta);
	w.y = r * sinf(theta);
	w.z = 1.0f - w.x * w.x - w.y * w.y;
	w.z = (0.0f < w.z) ? sqrtf(w.z) : 0.0f;

	pdf = w.z * M_1_PIf;

	// Align with axis.
	align_vector(axis, w);
}

RT_PROGRAM void closest_hit() {
	float3 geometry_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, var_geometry_normal));
	float3 normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, var_normal));

	the_payload.pos = the_ray.origin + the_ray.direction * the_intersection_distance;

	the_payload.flags |= (0.0f <= optix::dot(the_payload.wo, geometry_normal)) ? FLAG_FRONTFACE : 0;

	if ((the_payload.flags & FLAG_FRONTFACE) == 0) {
		// looking at backface
		geometry_normal = -geometry_normal;
		normal = -normal;
	}

	the_payload.radiance = optix::make_float3(0.0f);

	the_payload.f_over_pdf = optix::make_float3(0.0f);
	the_payload.pdf = 0.0f;

	unit_square_to_cosine_hemisphere(rng2(the_payload.seed), normal, the_payload.wi, the_payload.pdf);

	if (the_payload.pdf <= 0.0f || optix::dot(the_payload.wi, geometry_normal) <= 0.0f) {
		the_payload.flags |= FLAG_TERMINATE;
		return;
	}

	MaterialParameter parameters = sys_material_parameters[per_material_index];

	the_payload.f_over_pdf = parameters.albedo * (M_1_PIf * optix::dot(the_payload.wi, normal) / the_payload.pdf);
	the_payload.flags |= FLAG_DIFFUSE;

	if(sys_num_lights > 0) {
		const float2 sample = rng2(the_payload.seed);

		LightSample light_sample;

		light_sample.index = optix::clamp(static_cast<int>(floorf(rng(the_payload.seed) * sys_num_lights)), 0, sys_num_lights - 1);
		const LightType light_type = sys_light_definitions[light_sample.index].type;

		sys_sample_light[light_type](the_payload.pos, sample, light_sample);
		if(light_sample.pdf > 0.0f) {
			const float3 f = parameters.albedo * M_1_PIf;
			const float pdf = fmax(0.0f, optix::dot(light_sample.direction, normal) * M_1_PIf);

			if(pdf > 0.0f && isNotNull(f)) {
				ShadowRayPayload shadow_payload;
				shadow_payload.visible = true;

				optix::Ray ray = optix::make_Ray(the_payload.pos, light_sample.direction, 1, sys_scene_epsilon, light_sample.distance - sys_scene_epsilon);
				rtTrace(sys_top_object, ray, shadow_payload);

				if(shadow_payload.visible) {
					const float mis_weight = powerHeuristic(light_sample.pdf, pdf);
					the_payload.radiance += light_sample.emission * (mis_weight * optix::dot(light_sample.direction, normal) / light_sample.pdf);
				}
			}
		}

	}
}
