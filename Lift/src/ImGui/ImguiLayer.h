#pragma once

#include "Layer.h"

namespace lift {

	class ImGuiLayer : public Layer {
	public:
		ImGuiLayer();
		~ImGuiLayer();

		void OnAttach() override;
		void OnDetach() override;
		void OnImguiRender() override;

		static void Begin();
		static void End();
		bool show_demo_window_;
		bool show_another_window_;
		vec3 clear_color_ {0.f, 0.f, 0.f};
		
	};

}
