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
private:
    static ivec2 s_render_window_size;
    bool is_render_hovered_;
    bool show_render_window_ = true;
};

}
