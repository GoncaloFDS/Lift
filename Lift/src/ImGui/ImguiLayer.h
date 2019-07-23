#pragma once

#include "Layer.h"
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

		static void Begin();
		static void End();

		bool show_demo_window_;
		bool show_another_window_;

	};

}
