#pragma once

struct Light {
    float3 emission{1.0f, 1.0f, 1.0f};
    float3 position{};
	float3 v1{}, v2{};
	float3 normal{};
};

