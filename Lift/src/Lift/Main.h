#pragma once
#include "Application.h"

#ifdef LF_PLATFORM_WINDOWS

int main(int argc, char* argv[]) {

	lift::Log::Init();
	LF_CORE_INFO("Initialized Log!");

	auto app = lift::CreateApplication();
	app->Run();
	delete app;
	return 0;
}

#endif
