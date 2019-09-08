#include "pch.h"
#include "layer_stack.h"

lift::LayerStack::LayerStack() = default;

lift::LayerStack::~LayerStack() = default;

template<typename T>
void lift::LayerStack::popLayer(std::string name) {
    //	const auto it = std::find(layers_.begin(), layers_.end(), layer);
    //	if (it != layers_.end()) {
    //		layers_.erase(it);
    //		--layer_insert_index_;
    //	}
}

template<typename T>
void lift::LayerStack::popOverlay(std::string name) {
    //	const auto it = std::find(layers_.begin(), layers_.end(), overlay);
    //	if (it != layers_.end()) {
    //		layers_.erase(it);
    //	}
    //
}

std::vector<std::unique_ptr<lift::Layer>>::iterator lift::LayerStack::begin() {
    return layers_.begin();
}

std::vector<std::unique_ptr<lift::Layer>>::iterator lift::LayerStack::end() {
    return layers_.end();
}
