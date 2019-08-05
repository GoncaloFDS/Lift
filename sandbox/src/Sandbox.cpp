#include <Lift.h>

#include "imgui/imgui.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

class ExampleLayer : public lift::Layer {
public:
	ExampleLayer()
		: Layer("Example") {
	}

	void OnUpdate() override {
		if (lift::Input::IsKeyPressed(LF_KEY_TAB))
			LF_INFO("Tab pressed");
	}

	void OnEvent(lift::Event& event) override {
		if (event.GetEventType() == lift::EventType::KeyPressed) {
			auto& e = dynamic_cast<lift::KeyPressedEvent&>(event);
			if (e.GetKeyCode() == LF_KEY_TAB)
				LF_TRACE("Tab key is pressed (event)!");
			LF_TRACE("{0}", static_cast<char>(e.GetKeyCode()));
		}
	}

	void OnImguiRender() override {
	}
};

class Sandbox : public lift::Application {
public:
	Sandbox() {
		PushLayer<ExampleLayer>();
	}

	~Sandbox() = default;

};

std::shared_ptr<lift::Application> lift::CreateApplication() {
	return std::make_shared<Sandbox>();
}
