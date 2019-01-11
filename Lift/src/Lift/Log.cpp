#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

namespace Lift {

	std::shared_ptr<spdlog::logger> Log::sCoreLogger_;
	std::shared_ptr<spdlog::logger> Log::sClientLogger_;

	void Log::Init() {
		spdlog::set_pattern("%^[%T] %n: %v%$");
		sCoreLogger_ = spdlog::stdout_color_mt("LIFT");
		sCoreLogger_->set_level(spdlog::level::trace);
		
		sClientLogger_ = spdlog::stdout_color_mt("APP");
		sClientLogger_->set_level(spdlog::level::trace);
	}

}
