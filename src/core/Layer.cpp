﻿#include "pch.h"
#include "Layer.h"
#include <utility>

lift::Layer::Layer(std::string name)
	: name_(std::move(name)) {
}

lift::Layer::~Layer() {
	Layer::OnDetach();
}

const std::string& lift::Layer::GetName() const {
	return name_;
}