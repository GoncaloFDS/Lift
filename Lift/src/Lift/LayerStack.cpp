#include "pch.h"
#include "LayerStack.h"

Lift::LayerStack::LayerStack() {
	_layerInsert = _layers.begin();
}

Lift::LayerStack::~LayerStack() {
	for (Layer* layer : _layers) {
		delete layer;
	}
}

void Lift::LayerStack::PushLayer(Layer* layer) {
	_layerInsert = _layers.emplace(_layerInsert, layer);		
}

void Lift::LayerStack::PushOverlay(Layer* overlay) {
	_layers.emplace_back(overlay);
}

void Lift::LayerStack::PopLayer(Layer* layer) {
	const auto it = std::find(_layers.begin(), _layers.end(), layer);
	if (it != _layers.end()) {
		_layers.erase(it);
		--_layerInsert;
	}
}

void Lift::LayerStack::PopOverlay(Layer* overlay) {
	const auto it = std::find(_layers.begin(), _layers.end(), overlay);
	if (it != _layers.end()) {
		_layers.erase(it);
	}

}

std::vector<Lift::Layer*>::iterator Lift::LayerStack::begin() {
	return _layers.begin();
}

std::vector<Lift::Layer*>::iterator Lift::LayerStack::end() {
	return _layers.end();
}
