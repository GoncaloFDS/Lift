#include "pch.h"
#include "Layer.h"
#include <utility>

Lift::Layer::Layer(std::string name)
	: m_debugName(std::move(name)) {
}

Lift::Layer::~Layer() {
}

const std::string& Lift::Layer::GetName() const {
	return m_debugName;
}
