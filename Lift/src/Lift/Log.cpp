#include "pch.h"
#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

namespace Lift {

	std::shared_ptr<spdlog::logger> Log::_sCoreLogger;
	std::shared_ptr<spdlog::logger> Log::_sClientLogger;

	void Log::Init() {
		spdlog::set_pattern("%^[%T] %n: %v%$");
		_sCoreLogger = spdlog::stdout_color_mt("LIFT");
		_sCoreLogger->set_level(spdlog::level::trace);
		
		_sClientLogger = spdlog::stdout_color_mt("APP");
		_sClientLogger->set_level(spdlog::level::trace);
	}

}
