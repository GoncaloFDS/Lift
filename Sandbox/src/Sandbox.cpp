#include <Lift.h>

class ExampleLayer : public Lift::Layer {
public:
	ExampleLayer()
		: Layer("Example") {
	}

	void OnUpdate() override {
		if(Lift::Input::IsKeyPressed(LF_KEY_TAB))
			LF_INFO("Tab pressed");
	}

	void OnEvent(Lift::Event& event) override {
		//LF_TRACE("{0}", event);
	}
	
	void OnImGuiRender() override {
		
	}
};


class Sandbox : public Lift::Application {
public:
	Sandbox() {
		PushLayer(new ExampleLayer());
	}

	~Sandbox() {
		
	}

};

Lift::Application* Lift::CreateApplication() {
	return new Sandbox();
}


