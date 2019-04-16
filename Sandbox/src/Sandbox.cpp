#include <Lift.h>

class ExampleLayer : public Lift::Layer {
public:
	ExampleLayer()
		: Layer("Example") {
	}

	void OnUpdate() override {
		//LF_INFO("ExampleLayer::Update");
	}

	void OnEvent(Lift::Event& event) override {
		//LF_TRACE("{0}", event);
	}
};


class Sandbox : public Lift::Application {
public:
	Sandbox() {
		PushLayer(new ExampleLayer());
		PushOverlay(new Lift::ImGuiLayer());
	}

	~Sandbox() {
		
	}

};

Lift::Application* Lift::CreateApplication() {
	return new Sandbox();
}


