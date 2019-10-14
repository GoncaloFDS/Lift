#include "lift.h"

class ExampleLayer : public lift::Layer {
public:
    ExampleLayer()
        : Layer("Example") {
    }

    void onUpdate() override {
        if (lift::Input::isKeyPressed(LF_KEY_TAB))
            LF_INFO("Tab pressed");
    }

    void onEvent(lift::Event& event) override {
        if (event.getEventType() == lift::EventType::KEY_PRESSED) {
            auto& e = dynamic_cast<lift::KeyPressedEvent&>(event);
            if (e.getKeyCode() == LF_KEY_TAB)
                LF_TRACE("Tab key is pressed (event)!");
            //LF_TRACE("{0}", static_cast<char>(e.getKeyCode()));
        }
    }

    void onImguiRender() override {
    }
};

class Sandbox : public lift::Application {
public:
    Sandbox() {
        pushLayer<ExampleLayer>();
    }

    ~Sandbox() = default;

};

std::shared_ptr<lift::Application> lift::createApplication() {
    return std::make_shared<Sandbox>();
}
