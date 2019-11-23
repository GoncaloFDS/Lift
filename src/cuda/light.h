#pragma once

struct Light {
    enum class Falloff : int32_t {
        NONE = 0,
        LINEAR,
        QUADRATIC
    };

    enum class Type : int32_t {
        POINT = 0,
        PARALLELOGRAM
    } type;

    float3 color{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    float3 position{};
    Falloff falloff{Falloff::QUADRATIC};
	float3 v1{}, v2{};
	float3 normal{};
};

