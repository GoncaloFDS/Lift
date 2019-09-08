#include "pch.h"
#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> lift::Log::k_CoreLogger;

void lift::Log::init() {
    spdlog::set_pattern("[%n] %^%v%$");
    //spdlog::set_pattern("%^[%T] %n: %v%$");
    k_CoreLogger = spdlog::stdout_color_mt("lift");
    k_CoreLogger->set_level(spdlog::level::trace);

}
