#pragma once

#include "core.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"
#include <utility>

class Log {
  public:
  static void init();

  static auto getCoreLogger() -> std::shared_ptr<spdlog::logger> & { return s_logger_; }

  private:
  static std::shared_ptr<spdlog::logger> s_logger_;
};

// Core log macros
#define LF_TRACE(...) Log::getCoreLogger()->trace(__VA_ARGS__)
#define LF_INFO(...) Log::getCoreLogger()->info(__VA_ARGS__)
#define LF_WARN(...) Log::getCoreLogger()->warn(__VA_ARGS__)
#define LF_ERROR(...) Log::getCoreLogger()->error(__VA_ARGS__)
#define LF_FATAL(...) Log::getCoreLogger()->critical(__VA_ARGS__)
