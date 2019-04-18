#pragma once
#include "Application.h"

#ifdef LF_PLATFORM_WINDOWS

extern Lift::Application* Lift::CreateApplication();

int main(int argc, char* argv[]) {

	Lift::Log::Init();
	LF_CORE_INFO("Initialized Log!");

	auto app = Lift::CreateApplication();
	app->Run();
	delete app;
	return 0;
}

#endif
