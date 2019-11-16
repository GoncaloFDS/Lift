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

auto lift::LayerStack::begin() -> std::vector<std::unique_ptr<lift::Layer>>::iterator {
    return layers_.begin();
}

auto lift::LayerStack::end() -> std::vector<std::unique_ptr<lift::Layer>>::iterator {
    return layers_.end();
}
