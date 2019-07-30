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

		static void Begin();
		static void End();

	};

}
