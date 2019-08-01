#pragma once

#include "Core/Layer.h"
#include "Application.h"

namespace lift {

	class ImGuiLayer : public Layer {
	public:
		ImGuiLayer();
		~ImGuiLayer();

		void OnAttach() override;
		void OnDetach() override;
		void OnUpdate() override;
		void OnImguiRender() override;
		void OnEvent(Event& event) override;

		static vec2 GetRenderWindowSize();
		static void Begin();
		static void End();

		static vec2 render_window_size_;
		unsigned render_id_;
		bool is_render_hovered_;
	};

}
