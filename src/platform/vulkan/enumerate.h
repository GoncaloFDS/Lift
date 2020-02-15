#pragma once

#include "core/utilities.h"
#include <vector>

namespace vulkan {

template<class TValue>
inline std::vector<TValue> getEnumerateVector(VkResult(enumerate)(uint32_t *, TValue *), std::vector<TValue> &vector) {
    uint32_t count = 0;
    vulkanCheck(enumerate(&count, nullptr), "enumerate");

    vector.resize(count);
    vulkanCheck(enumerate(&count, vector.data()), "enumerate");

    return vector;
}

template<class THandle, class TValue>
inline std::vector<TValue>
getEnumerateVector(THandle handle, void(enumerate)(THandle, uint32_t *, TValue *), std::vector<TValue> &vector) {
    uint32_t count = 0;
    enumerate(handle, &count, nullptr);

    vector.resize(count);
    enumerate(handle, &count, vector.data());

    return vector;
}

template<class THandle, class TValue>
inline std::vector<TValue>
getEnumerateVector(THandle handle, VkResult(enumerate)(THandle, uint32_t *, TValue *), std::vector<TValue> &vector) {
    uint32_t count = 0;
    vulkanCheck(enumerate(handle, &count, nullptr), "enumerate");

    vector.resize(count);
    vulkanCheck(enumerate(handle, &count, vector.data()), "enumerate");

    return vector;
}

template<class THandle1, class THandle2, class TValue>
inline std::vector<TValue> getEnumerateVector(THandle1 handle_1,
                                              THandle2 handle_2,
                                              VkResult(enumerate)(THandle1, THandle2, uint32_t *, TValue *),
                                              std::vector<TValue> &vector) {
    uint32_t count = 0;
    vulkanCheck(enumerate(handle_1, handle_2, &count, nullptr), "enumerate");

    vector.resize(count);
    vulkanCheck(enumerate(handle_1, handle_2, &count, vector.data()), "enumerate");

    return vector;
}

template<class TValue>
inline std::vector<TValue> getEnumerateVector(VkResult(enumerate)(uint32_t *, TValue *)) {
    std::vector<TValue> initial;
    return getEnumerateVector(enumerate, initial);
}

template<class THandle, class TValue>
inline std::vector<TValue> getEnumerateVector(THandle handle, void(enumerate)(THandle, uint32_t *, TValue *)) {
    std::vector<TValue> initial;
    return getEnumerateVector(handle, enumerate, initial);
}

template<class THandle, class TValue>
inline std::vector<TValue> getEnumerateVector(THandle handle, VkResult(enumerate)(THandle, uint32_t *, TValue *)) {
    std::vector<TValue> initial;
    return getEnumerateVector(handle, enumerate, initial);
}

template<class THandle1, class THandle2, class TValue>
inline std::vector<TValue> getEnumerateVector(THandle1 handle_1,
                                              THandle2 handle_2,
                                              VkResult(enumerate)(THandle1, THandle2, uint32_t *, TValue *)) {
    std::vector<TValue> initial;
    return getEnumerateVector(handle_1, handle_2, enumerate, initial);
}

}  // namespace vulkan
