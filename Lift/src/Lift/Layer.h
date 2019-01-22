#pragma once

#include "Core.h"
#include "Events/Event.h"


namespace Lift {
	
	class LIFT_API Layer {
	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnEvent(Event& event) {}

		inline const std::string& GetName() const;

	protected:
		std::string _debugName;
	};
}
