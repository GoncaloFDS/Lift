#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 5) rayPayloadInEXT bool is_shadowed_;

void main() {
    is_shadowed_ = false;
}