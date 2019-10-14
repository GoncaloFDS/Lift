#include "pch.h"
#include "log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> lift::Log::s_logger;

void lift::Log::init() {
    spdlog::set_pattern("[%n] %^%v%$");
    //spdlog::set_pattern("%^[%T] %n: %v%$");
    s_logger = spdlog::stdout_color_mt("lift");
    s_logger->set_level(spdlog::level::trace);

}
