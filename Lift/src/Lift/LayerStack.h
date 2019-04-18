#pragma once

#include "Core.h"
#include "Layer.h"

namespace Lift {

	class LayerStack {
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
		unsigned int m_layerInsertIndex = 0;
	};
}
