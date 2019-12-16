#pragma once
#include "core.h"
#include "core/io/log.h"
#include <cuda.h>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include "cuda_gl_interop.h"

namespace lift {

enum class CudaOutputBufferType {
    CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
    GL_INTEROP = 1, // single device only, preferred for single device
    ZERO_COPY = 2, // general case, preferred for multi-gpu if not fully nvlink connected
    CUDA_P2P = 3  // fully connected only, preferred for fully nvlink connected
};

template<typename PIXEL_FORMAT>
class CudaOutputBuffer {
public:
    CudaOutputBuffer(CudaOutputBufferType type, int32_t width, int32_t height);
    ~CudaOutputBuffer();

    void setDevice(int32_t device_idx) { device_idx_ = device_idx; }
    void setStream(CUstream stream) { stream_ = stream; }

    void resize(int32_t width, int32_t height);

    // Allocate or update device pointer as necessary for CUDA access
    auto map() -> PIXEL_FORMAT*;
    void unmap();

    auto width() -> int32_t { return width_; }
    auto height() -> int32_t { return height_; }

    // Get output buffer
    auto getPixelBufferObject() -> GLuint;
    auto getHostPointer() -> PIXEL_FORMAT*;

private:
    void makeCurrent() { CUDA_CHECK(cudaSetDevice(device_idx_)); }

    CudaOutputBufferType m_type;

    int32_t width_ = 0u;
    int32_t height_ = 0u;

    cudaGraphicsResource* cuda_gfx_resource_ = nullptr;
    GLuint pbo_ = 0u;
    PIXEL_FORMAT* device_pixels_ = nullptr;
    PIXEL_FORMAT* host_copy_pixels_ = nullptr;
    std::vector<PIXEL_FORMAT> host_pixels_;

    CUstream stream_ = nullptr;
    int32_t device_idx_ = 0;
};

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::CudaOutputBuffer(CudaOutputBufferType type, int32_t width, int32_t height)
    : m_type(type) {
    // If using GL Interop, expect that the active device is also the display device.
    if (type == CudaOutputBufferType::GL_INTEROP) {
        int current_device, is_display_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
    }
    resize(width, height);
}

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::~CudaOutputBuffer() {
    try {
        makeCurrent();
        if (m_type == CudaOutputBufferType::CUDA_DEVICE || m_type == CudaOutputBufferType::CUDA_P2P) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( device_pixels_ )));
        } else if (m_type == CudaOutputBufferType::ZERO_COPY) {
            CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>( host_copy_pixels_ )));
        } else if (m_type == CudaOutputBufferType::GL_INTEROP) {
            // nothing needed
        }

        if (pbo_ != 0u) {
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            GL_CHECK(glDeleteBuffers(1, &pbo_));
        }
    }
    catch (std::exception& e) {
        std::cerr << "CudaOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::resize(int32_t width, int32_t height) {
    if (width_ == width && height_ == height)
        return;

    width_ = width;
    height_ = height;

    makeCurrent();

    if (m_type == CudaOutputBufferType::CUDA_DEVICE || m_type == CudaOutputBufferType::CUDA_P2P) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>( device_pixels_ )));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>( &device_pixels_ ),
            width_ * height_ * sizeof(PIXEL_FORMAT)
        ));

    }

    if (m_type == CudaOutputBufferType::GL_INTEROP || m_type == CudaOutputBufferType::CUDA_P2P) {
        // GL buffer gets resized below
        GL_CHECK(glGenBuffers(1, &pbo_));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * width * height, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &cuda_gfx_resource_,
            pbo_,
            cudaGraphicsMapFlagsWriteDiscard
        ));
    }

    if (m_type == CudaOutputBufferType::ZERO_COPY) {
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>( host_copy_pixels_ )));
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>( &host_copy_pixels_ ),
            width_ * height_ * sizeof(PIXEL_FORMAT),
            cudaHostAllocPortable | cudaHostAllocMapped
        ));
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>( &device_pixels_ ),
            reinterpret_cast<void*>( host_copy_pixels_ ),
            0 /*flags*/
        ));
    }

    if (m_type != CudaOutputBufferType::GL_INTEROP && m_type != CudaOutputBufferType::CUDA_P2P && pbo_ != 0u) {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * width * height, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));
    }

    if (!host_pixels_.empty())
        host_pixels_.resize(width_ * height_);
}

template<typename PIXEL_FORMAT>
auto CudaOutputBuffer<PIXEL_FORMAT>::map() -> PIXEL_FORMAT* {
    if (m_type == CudaOutputBufferType::CUDA_DEVICE || m_type == CudaOutputBufferType::CUDA_P2P) {
        // nothing needed
    } else if (m_type == CudaOutputBufferType::GL_INTEROP) {
        makeCurrent();

        size_t buffer_size = 0u;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_gfx_resource_, stream_));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>( &device_pixels_ ),
            &buffer_size,
            cuda_gfx_resource_
        ));
    } else // m_type == CudaOutputBufferType::ZERO_COPY
    {
        // nothing needed
    }

    return device_pixels_;
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::unmap() {
    makeCurrent();

    if (m_type == CudaOutputBufferType::CUDA_DEVICE || m_type == CudaOutputBufferType::CUDA_P2P) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else if (m_type == CudaOutputBufferType::GL_INTEROP) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_gfx_resource_, stream_));
    } else // m_type == CudaOutputBufferType::ZERO_COPY
    {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

template<typename PIXEL_FORMAT>
auto CudaOutputBuffer<PIXEL_FORMAT>::getPixelBufferObject() -> GLuint {
    if (pbo_ == 0u)
        GL_CHECK(glGenBuffers(1, &pbo_));

    const size_t buffer_size = width_ * height_ * sizeof(PIXEL_FORMAT);

    if (m_type == CudaOutputBufferType::CUDA_DEVICE) {
        // We need a host buffer to act as a way-station
        if (host_pixels_.empty())
            host_pixels_.resize(width_ * height_);

        makeCurrent();
        CUDA_CHECK(cudaMemcpy(
            static_cast<void*>( host_pixels_.data()),
            device_pixels_,
            buffer_size,
            cudaMemcpyDeviceToHost
        ));

        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
        GL_CHECK(glBufferData(
            GL_ARRAY_BUFFER,
            buffer_size,
            static_cast<void*>( host_pixels_.data()),
            GL_STREAM_DRAW
        ));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    } else if (m_type == CudaOutputBufferType::GL_INTEROP) {
        // Nothing needed
    } else if (m_type == CudaOutputBufferType::CUDA_P2P) {
        makeCurrent();
        void* pbo_buff = nullptr;
        size_t dummy_size = 0;

        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_gfx_resource_, stream_));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pbo_buff, &dummy_size, cuda_gfx_resource_));
        CUDA_CHECK(cudaMemcpy(pbo_buff, device_pixels_, buffer_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_gfx_resource_, stream_));
    } else // m_type == CudaOutputBufferType::ZERO_COPY
    {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
        GL_CHECK(glBufferData(
            GL_ARRAY_BUFFER,
            buffer_size,
            static_cast<void*>( host_copy_pixels_ ),
            GL_STREAM_DRAW
        ));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    return pbo_;
}

template<typename PIXEL_FORMAT>
auto CudaOutputBuffer<PIXEL_FORMAT>::getHostPointer() -> PIXEL_FORMAT* {
    if (m_type == CudaOutputBufferType::CUDA_DEVICE ||
        m_type == CudaOutputBufferType::CUDA_P2P ||
        m_type == CudaOutputBufferType::GL_INTEROP) {
        host_pixels_.resize(width_ * height_);

        makeCurrent();
        CUDA_CHECK(cudaMemcpy(
            static_cast<void*>( host_pixels_.data()),
            map(),
            width_ * height_ * sizeof(PIXEL_FORMAT),
            cudaMemcpyDeviceToHost
        ));
        unmap();

        return host_pixels_.data();
    } else // m_type == CudaOutputBufferType::ZERO_COPY
    {
        return host_copy_pixels_;
    }
}

}


