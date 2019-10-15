#pragma once

#include "core/layer.h"
#include "application.h"

namespace lift {

class ImGuiLayer : public Layer {
public:
    ImGuiLayer();
    ~ImGuiLayer();

    void onAttach() override;
    void onDetach() override;
    void onUpdate() override;
    void onImguiRender() override;
    void onEvent(Event& event) override;

    static void begin();
    static void end();
};

}
