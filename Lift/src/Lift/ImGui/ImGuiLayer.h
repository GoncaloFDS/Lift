#pragma once

#include "Lift/Layer.h"

namespace Lift {
	

	class LIFT_API ImGuiLayer : public Layer {
	public:
		ImGuiLayer();
		~ImGuiLayer();

		void OnAttach() override;
		void OnDetach() override;
		void OnUpdate() override;
		void OnImGuiRender() override;

		void Begin();
		void End();
	private:
		float m_time = 0.0f;
	};

}