#pragma once
#include "application.h"

//#ifdef LF_PLATFORM_WINDOWS

auto main(int argc, char* argv[]) -> int {
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    lift::Log::init();
    LF_INFO("Initialized Log!");

    std::shared_ptr<lift::Application> app = lift::createApplication();

    app->run();

    return 0;
}

//#endif
