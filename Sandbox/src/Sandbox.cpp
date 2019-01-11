#include <Lift.h>

class Sandbox : public Lift::Application {
public:
	Sandbox() {
		
	}

	~Sandbox() {
		
	}

};

Lift::Application* Lift::CreateApplication() {
	return new Sandbox();
}


