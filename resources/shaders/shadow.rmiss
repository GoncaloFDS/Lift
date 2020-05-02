#version 460
#extension GL_NV_ray_tracing : require

layout(location = 5) rayPayloadInNV bool is_shadowed_;

void main() {
    is_shadowed_ = false;
}