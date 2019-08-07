#pragma once
#include "Core.h"
#include "core/io/Log.h"
#include <cuda.h>
#include <cuda_runtime.h>


namespace lift {
	struct CudaBuffer {
		[[nodiscard]] CUdeviceptr d_pointer() const { return CUdeviceptr(d_ptr); }
		//! re-size buffer to given number of bytes
		void resize(const size_t size) {
			if (d_ptr) free();
			alloc(size);
		}

		//! allocate to given number of bytes
		void alloc(size_t size) {
			LF_CORE_ASSERT(d_ptr == nullptr, " ");
			this->size_in_bytes = size;
			CUDA_CHECK(Malloc(static_cast<void**>(&d_ptr), size_in_bytes));
		}

		//! free allocated memory
		void free() {
			CUDA_CHECK(Free(d_ptr));
			d_ptr = nullptr;
			size_in_bytes = 0;
		}

		template <typename T>
		void alloc_and_upload(const std::vector<T>& vt) {
			alloc(vt.size() * sizeof(T));
			upload(static_cast<const T*>(vt.data()), vt.size());
		}

		template <typename T>
		void upload(const T* t, size_t count) {
			LF_CORE_ASSERT(d_ptr != nullptr, " ");
			LF_CORE_ASSERT(size_in_bytes == count*sizeof(T), " ");
			CUDA_CHECK(Memcpy(d_ptr, (void *)t,
							  count * sizeof(T), cudaMemcpyHostToDevice));
		}

		template <typename T>
		void download(T* t, size_t count) {
			LF_CORE_ASSERT(d_ptr != nullptr, " ");
			LF_CORE_ASSERT(size_in_bytes == count*sizeof(T), " ");
			CUDA_CHECK(Memcpy((void *)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		size_t size_in_bytes{0};
		void* d_ptr{nullptr};
	};
}
