#include "pch.h"
#include "LayerStack.h"

lift::LayerStack::LayerStack() {
}

lift::LayerStack::~LayerStack() {
	for(Layer* layer : layers_) {
		delete layer;
	}
}

void lift::LayerStack::PushLayer(Layer* layer) {
	layers_.emplace(layers_.begin() + layer_insert_index_++, layer);
}

void lift::LayerStack::PushOverlay(Layer* overlay) {
	layers_.emplace_back(overlay);
}

void lift::LayerStack::PopLayer(Layer* layer) {
	const auto it = std::find(layers_.begin(), layers_.end(), layer);
	if(it != layers_.end()) {
		layers_.erase(it);
		--layer_insert_index_;
	}
}

void lift::LayerStack::PopOverlay(Layer* overlay) {
	const auto it = std::find(layers_.begin(), layers_.end(), overlay);
	if(it != layers_.end()) {
		layers_.erase(it);
	}

}

std::vector<lift::Layer*>::iterator lift::LayerStack::begin() {
	return layers_.begin();
}

std::vector<lift::Layer*>::iterator lift::LayerStack::end() {
	return layers_.end();
}
