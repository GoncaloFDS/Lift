#include "pch.h"
#include "Renderer.h"
#include "RenderCommand.h"

void lift::Renderer::BeginScene() {

}

void lift::Renderer::EndScene() {
}

void lift::Renderer::Submit(const std::shared_ptr<VertexArray>& vertex_array) {
	vertex_array->Bind();
	RenderCommand::DrawIndexed(vertex_array);
}
