#pragma once

#include "Core.h"
#include "Layer.h"

namespace lift {

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
		std::vector<Layer*> layers_;
		unsigned int layer_insert_index_ = 0;
	};
}
