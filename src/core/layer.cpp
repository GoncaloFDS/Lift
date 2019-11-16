#include "pch.h"
#include "layer.h"
#include <utility>

lift::Layer::Layer(std::string name)
    : name_(std::move(name)) {
}

lift::Layer::~Layer() {
    Layer::onDetach();
}

auto lift::Layer::getName() const -> const std::string& {
    return name_;
}

void lift::Layer::onAttach() {
}

void lift::Layer::onDetach() {
}

void lift::Layer::onUpdate() {
}

void lift::Layer::onImguiRender() {
}

void lift::Layer::onEvent(lift::Event& event) {
}
