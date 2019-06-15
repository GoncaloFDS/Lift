#version 330 core

layout(location = 0) out vec4 color;

in vec3 v_Position;
in vec4 v_Color;

void main()	{
	color = vec4(v_Position * 0.5 + 0.5, 1.0);
	color = v_Color;
}
