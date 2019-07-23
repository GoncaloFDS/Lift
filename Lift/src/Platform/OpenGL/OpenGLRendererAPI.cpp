#include "pch.h"
#include "OpenGLRendererAPI.h"

#include <glad/glad.h>
#include <glm/common.hpp>

void lift::OpenGLRendererAPI::SetClearColor(const vec4& color) {
	glClearColor(color.x, color.y, color.z, color.w);
}

void lift::OpenGLRendererAPI::Clear() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void lift::OpenGLRendererAPI::DrawIndexed(const std::shared_ptr<VertexArray>& vertex_array) {
	glDrawElements(GL_TRIANGLES, vertex_array->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
}

void lift::OpenGLRendererAPI::Resize(uint32_t width, uint32_t height) {
	glViewport(0, 0, width, height);
}
