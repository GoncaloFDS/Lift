
#include "profiler.h"
#include "log.h"
#include <utility>

std::map<Profiler::Id, float> Profiler::timers_;

Profiler::Profiler(Id id) : duration_(0), id_(id) {
    start_ = std::chrono::high_resolution_clock::now();
}

Profiler::Profiler(std::string id) : duration_(0), message_(std::move(id)), id_(Id::Other) {
    start_ = std::chrono::high_resolution_clock::now();
}

Profiler::~Profiler() {
    end_ = std::chrono::high_resolution_clock::now();
    duration_ = end_ - start_;

    const auto s = duration_.count();
    if (!message_.empty()) {
        LF_INFO("{0} -> {1}s", message_, s);
    }
    timers_[id_] = s;
}

auto Profiler::getDuration(Id id) -> float {
    return timers_[id];
}

void Profiler::reset(Id id) {
    timers_[id] = 0.0f;
}
