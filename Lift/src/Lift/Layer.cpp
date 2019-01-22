#include "pch.h"
#include "Layer.h"

Lift::Layer::Layer(const std::string& name) 
	: _debugName(name) {
}

const std::string& Lift::Layer::GetName() const {
	return _debugName;
}
