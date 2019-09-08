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

    static ivec2 getRenderWindowSize();
    static void begin();
    static void end();

    static ivec2 render_window_size;
    unsigned render_id;
    bool is_render_hovered;
};

}
