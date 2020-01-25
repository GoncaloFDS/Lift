#pragma once

#include "core.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include <utility>

namespace lift {

class Log {
public:
    static void init();

    static auto getCoreLogger() -> std::shared_ptr<spdlog::logger>& { return s_logger_; }

private:
    static std::shared_ptr<spdlog::logger> s_logger_;
};

}

// Core log macros
#define LF_TRACE(...)    lift::Log::getCoreLogger()->trace(__VA_ARGS__)
#define LF_INFO(...)    lift::Log::getCoreLogger()->info(__VA_ARGS__)
#define LF_WARN(...)    lift::Log::getCoreLogger()->warn(__VA_ARGS__)
#define LF_ERROR(...)    lift::Log::getCoreLogger()->error(__VA_ARGS__)
#define LF_FATAL(...)    lift::Log::getCoreLogger()->critical(__VA_ARGS__)

