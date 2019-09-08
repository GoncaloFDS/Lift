#include "pch.h"
#include "layer.h"
#include <utility>

lift::Layer::Layer(std::string name)
    : name_(std::move(name)) {
}

lift::Layer::~Layer() {
    Layer::onDetach();
}

const std::string& lift::Layer::getName() const {
    return name_;
}
