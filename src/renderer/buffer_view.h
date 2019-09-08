#pragma once
#include <cstdint>
#include <cuda/preprocessor.h>

namespace lift {

template<typename T>
struct BufferView {
    CUdeviceptr data = 0;
    uint32_t count = 0;
    uint16_t byte_stride = 0;
    uint16_t elmt_byte_size = 0;

    SUTIL_HOSTDEVICE [[nodiscard]] bool isValid() const { return static_cast<bool>(data); }

    SUTIL_HOSTDEVICE explicit operator bool() const { return isValid(); }

    SUTIL_HOSTDEVICE const T &operator[](uint32_t idx) const {
        return *reinterpret_cast<T *>(data + idx * (byte_stride ? byte_stride : sizeof(T)));
    }
};

typedef BufferView<uint32_t> GenericBufferView;

}
