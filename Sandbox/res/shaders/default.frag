#version 330 core

layout(location = 0) out vec4 color;

in vec3 v_Position;
in vec4 v_Color;
in vec2 v_Texture_coord;

uniform sampler2D u_Texture;

void main()	{
	vec4 texture_color = texture(u_Texture, v_Texture_coord);
	color = texture_color;
}
