
#include "log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> Log::s_logger_;

void Log::init() {
     spdlog::set_pattern("%^[%T] %n: %v%$");
    s_logger_ = spdlog::stdout_color_mt("lift");
    s_logger_->set_level(spdlog::level::debug);
    LF_INFO("Initialized Log");
}
