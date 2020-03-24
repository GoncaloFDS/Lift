#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "material.glsl"

layout(binding = 0) uniform sampler2D out_texture;

layout(location = 0) in vec3 FragColor;
layout(location = 1) in vec3 FragNormal;
layout(location = 2) in vec2 FragTexCoord;
layout(location = 3) in flat int FragMaterialIndex;

layout(location = 0) out vec4 OutColor;

void main() {
  vec3 t = texture(out_texture, FragTexCoord).rgb;

//  OutColor = vec4(0.9, 0.1, 0.5, 1);
  OutColor = vec4(t, 1.0);
}