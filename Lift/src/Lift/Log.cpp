#include "pch.h"
#include "Log.h"

#include "spdlog/sinks/stdout_color_sinks.h"

namespace Lift {

	std::shared_ptr<spdlog::logger> Log::m_sCoreLogger;
	std::shared_ptr<spdlog::logger> Log::m_sClientLogger;

	void Log::Init() {
		spdlog::set_pattern("%^[%T] %n: %v%$");
		m_sCoreLogger = spdlog::stdout_color_mt("Lift");
		m_sCoreLogger->set_level(spdlog::level::trace);
		
		m_sClientLogger = spdlog::stdout_color_mt("App");
		m_sClientLogger->set_level(spdlog::level::trace);
	}

	std::shared_ptr<spdlog::logger>& Log::GetCoreLogger() {
		return m_sCoreLogger;
	}

	std::shared_ptr<spdlog::logger>& Log::GetClientLogger() {
		return m_sClientLogger;
	}

}
