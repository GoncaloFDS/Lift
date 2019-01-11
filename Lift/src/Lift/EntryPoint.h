#pragma once
#include "Application.h"

#ifdef LF_PLATFORM_WINDOWS

extern Lift::Application* Lift::CreateApplication();

int main(int argc, char* argv[]) {
	
	Lift::Log::Init();
	LF_CORE_WARN("Initialized Log!");
	int a = 5;
	LF_INFO("Info Log! Var={0}", a);
	
	
	
	auto app = Lift::CreateApplication();
	app->Run();
	delete app;
	return 0;
}

#endif

