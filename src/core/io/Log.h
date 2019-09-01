#pragma once

#include "Core.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

namespace lift {

	class Log {
	public:
		static void Init();

		static std::shared_ptr<spdlog::logger>& GetCoreLogger() {
			return core_logger_;
		}

		static std::shared_ptr<spdlog::logger>& GetClientLogger() {
			return client_logger_;
		}

	private:
		static std::shared_ptr<spdlog::logger> core_logger_;
		static std::shared_ptr<spdlog::logger> client_logger_;
	};

}

// Core log macros
#define LF_CORE_TRACE(...)	lift::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define LF_CORE_INFO(...)	lift::Log::GetCoreLogger()->info(__VA_ARGS__)
#define LF_CORE_WARN(...)	lift::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LF_CORE_ERROR(...)	lift::Log::GetCoreLogger()->error(__VA_ARGS__)
#define LF_CORE_FATAL(...)	lift::Log::GetCoreLogger()->critical(__VA_ARGS__)

// Client log macros
#define LF_TRACE(...)	lift::Log::GetClientLogger()->trace(__VA_ARGS__)
#define LF_INFO(...)	lift::Log::GetClientLogger()->info(__VA_ARGS__)
#define LF_WARN(...)	lift::Log::GetClientLogger()->warn(__VA_ARGS__)
#define LF_ERROR(...)	lift::Log::GetClientLogger()->error(__VA_ARGS__)
#define LF_FATAL(...)	lift::Log::GetClientLogger()->critical(__VA_ARGS__)
