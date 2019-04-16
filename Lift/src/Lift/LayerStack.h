#pragma once

#include "Core.h"
#include "Layer.h"

namespace Lift {
	
	class LIFT_API LayerStack {
	public:
		LayerStack();
		~LayerStack();

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* overlay);
		void PopLayer(Layer* layer);
		void PopOverlay(Layer* overlay);
	
		std::vector<Layer*>::iterator begin();	
		std::vector<Layer*>::iterator end();	
	private:
		std::vector<Layer*> m_layers;
		std::vector<Layer*>::iterator m_layerInsert;
	};
}
