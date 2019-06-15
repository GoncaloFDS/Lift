#pragma once

#include "Lift/Layer.h"

namespace lift {

	class ImguiLayer : public Layer {
	public:
		ImguiLayer();
		~ImguiLayer();

		void OnAttach() override;
		void OnDetach() override;
		void OnImguiRender() override;

		static void Begin();
		static void End();
		bool show_demo_window_;
		bool show_another_window_;
		int clear_color_;
	};

}
