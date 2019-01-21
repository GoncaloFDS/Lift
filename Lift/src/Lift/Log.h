#pragma once

#include "Core.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

namespace Lift {
	
	class LIFT_API Log {
	public:
		static void Init();

		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return _sCoreLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return _sClientLogger; }
	private:
		static std::shared_ptr<spdlog::logger> _sCoreLogger;
		static std::shared_ptr<spdlog::logger> _sClientLogger;
	};

}


// Core log macros
#define LF_CORE_TRACE(...)	::Lift::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define LF_CORE_INFO(...)	::Lift::Log::GetCoreLogger()->info(__VA_ARGS__)
#define LF_CORE_WARN(...)	::Lift::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LF_CORE_ERROR(...)	::Lift::Log::GetCoreLogger()->error(__VA_ARGS__)
#define LF_CORE_FATAL(...)	::Lift::Log::GetCoreLogger()->fatal(__VA_ARGS__)

// Client log macros
#define LF_TRACE(...)	::Lift::Log::GetClientLogger()->trace(__VA_ARGS__)
#define LF_INFO(...)	::Lift::Log::GetClientLogger()->info(__VA_ARGS__)
#define LF_WARN(...)	::Lift::Log::GetClientLogger()->warn(__VA_ARGS__)
#define LF_ERROR(...)	::Lift::Log::GetClientLogger()->error(__VA_ARGS__)
#define LF_FATAL(...)	::Lift::Log::GetClientLogger()->fatal(__VA_ARGS__)