#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

enum LightType {
	LIGHT_ENVIRONMENT = 0,
	LIGHT_PARALLELOGRAM = 1
};

struct LightDefinition {
	LightType type;

	optix::float3 position;
	optix::float3 vector_u;
	optix::float3 vector_v;
	optix::float3 normal;
	optix::float3 emission;
	float area;

	//! Manual padding to float4 alignment goes here.
	float unused0;
	float unused1;
	float unused2;
};

struct LightSample {
	optix::float3 position;
	int index;
	optix::float3 direction;
	float distance;
	optix::float3 emission;
	float pdf;
};
