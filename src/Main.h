#pragma once
#include "Application.h"

//#ifdef LF_PLATFORM_WINDOWS

int main(int argc, char *argv[]) {
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    lift::Log::init();
    LF_INFO("Initialized Log!");

    std::shared_ptr<lift::Application> app = lift::createApplication();

    app->run();
    int *a = new int(1);

    return 0;
}

//#endif
