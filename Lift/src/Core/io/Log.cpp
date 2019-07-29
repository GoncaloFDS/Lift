#include "pch.h"
#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> lift::Log::core_logger_;
std::shared_ptr<spdlog::logger> lift::Log::client_logger_;

void lift::Log::Init() {
	spdlog::set_pattern("%^[%T] %n: %v%$");
	core_logger_ = spdlog::stdout_color_mt("lift");
	core_logger_->set_level(spdlog::level::trace);

	client_logger_ = spdlog::stdout_color_mt("App");
	client_logger_->set_level(spdlog::level::trace);
}
