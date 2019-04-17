#pragma once

#include "Lift/Layer.h"
#include "Lift/Events/MouseEvent.h"
#include "Lift/Events/KeyEvent.h"
#include "Lift/Events/ApplicationEvent.h"

namespace Lift {
	

	class LIFT_API ImGuiLayer : public Layer {
	public:
		ImGuiLayer();
		~ImGuiLayer();

		void OnAttach() override;
		void OnDetach() override;
		void OnImGuiRender() override;

		void Begin();
		void End();
	};

}
