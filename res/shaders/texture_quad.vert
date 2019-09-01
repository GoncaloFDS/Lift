#version 330 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_Uv;

out vec3 v_Position;
out vec2 v_Texture_coord;

void main() {
	v_Position = a_Position;
	v_Texture_coord = a_Uv;
	gl_Position = vec4(a_Position, 1.0);
}
