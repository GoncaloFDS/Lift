#include <Lift.h>

#include "imgui/imgui.h"

class ExampleLayer : public lift::Layer {
public:
	ExampleLayer()
		: Layer("Example") {
	}

	void OnUpdate() override {
		if(lift::Input::IsKeyPressed(LF_KEY_TAB))
			LF_INFO("Tab pressed");
	}

	void OnEvent(lift::Event& event) override {
		if(event.GetEventType() == lift::EventType::kKeyPressed) {
			auto& e = dynamic_cast<lift::KeyPressedEvent&>(event);
			if(e.GetKeyCode() == LF_KEY_TAB)
				LF_TRACE("Tab key is pressed (event)!");
			LF_TRACE("{0}", static_cast<char>(e.GetKeyCode()));
		}
	}

	void OnImGuiRender() override {
		ImGui::Begin("Test");
		ImGui::Text("Hello World!");
		ImGui::End();
	}
};

class Sandbox : public lift::Application {
public:
	Sandbox() {
		PushLayer(new ExampleLayer());
	}

	~Sandbox() = default;

};

lift::Application* lift::CreateApplication() {
	return new Sandbox();
}
