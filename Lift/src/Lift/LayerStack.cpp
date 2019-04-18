#include "pch.h"
#include "LayerStack.h"

Lift::LayerStack::LayerStack() {
}

Lift::LayerStack::~LayerStack() {
	for(Layer* layer : m_layers) {
		delete layer;
	}
}

void Lift::LayerStack::PushLayer(Layer* layer) {
	m_layers.emplace(m_layers.begin() + m_layerInsertIndex++, layer);
}

void Lift::LayerStack::PushOverlay(Layer* overlay) {
	m_layers.emplace_back(overlay);
}

void Lift::LayerStack::PopLayer(Layer* layer) {
	const auto it = std::find(m_layers.begin(), m_layers.end(), layer);
	if(it != m_layers.end()) {
		m_layers.erase(it);
		--m_layerInsertIndex;
	}
}

void Lift::LayerStack::PopOverlay(Layer* overlay) {
	const auto it = std::find(m_layers.begin(), m_layers.end(), overlay);
	if(it != m_layers.end()) {
		m_layers.erase(it);
	}

}

std::vector<Lift::Layer*>::iterator Lift::LayerStack::begin() {
	return m_layers.begin();
}

std::vector<Lift::Layer*>::iterator Lift::LayerStack::end() {
	return m_layers.end();
}
