#version 460

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D output_texture;


void main() {
    vec2 uv = outUV;
    fragColor = texture(output_texture, uv).rgba;
}