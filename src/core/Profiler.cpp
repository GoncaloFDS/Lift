#include "pch.h"
#include "Profiler.h"
#include <utility>

Profiler::Profiler(std::string message) : duration_(0), message_(std::move(message)) {
    start_ = std::chrono::high_resolution_clock::now();
}

Profiler::~Profiler() {
    end_ = std::chrono::high_resolution_clock::now();
    duration_ = end_ - start_;

    const auto s = duration_.count();
    if (message_.empty())
        LF_WARN("Function took {0}s", s);
    else
        LF_WARN("{0} -> {1}s", message_, s);
}
