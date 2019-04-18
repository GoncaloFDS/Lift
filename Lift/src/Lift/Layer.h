#pragma once

#include "Core.h"
#include "Events/Event.h"


namespace Lift {
	
	class Layer {
	public:
		Layer(std::string name = "Layer");
		virtual ~Layer();

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnImGuiRender() {}
		virtual void OnEvent(Event& event) {}

		inline const std::string& GetName() const;

	protected:
		std::string m_debugName;
	};
}
