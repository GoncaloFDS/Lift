#pragma once
#include "Core.h"
#include "core/io/Log.h"
#include <cuda.h>
#include <cuda_runtime.h>


namespace lift {
	template <typename T = char>
	struct CudaBuffer {
		CudaBuffer(const size_t count = 0) { alloc(count); }
		~CudaBuffer() { free(); }

		void alloc(const size_t count) {
			free();
			alloc_count_ = count_ = count;
			if (count_) {
				CUDA_CHECK(cudaMalloc( &ptr_, alloc_count_ * sizeof( T ) ));
			}
		}

		void allocIfRequired(size_t count) {
			if (count <= count_) {
				count_ = count;
				return;
			}
			alloc(count);
		}

		[[nodiscard]] CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>(ptr_); }
		[[nodiscard]] CUdeviceptr get(size_t index) const { return reinterpret_cast<CUdeviceptr>(ptr_ + index); }

		void free() {
			count_ = 0;
			alloc_count_ = 0;
			CUDA_CHECK(cudaFree( ptr_ ));
			ptr_ = nullptr;
		}

		CUdeviceptr release() {
			const auto current = reinterpret_cast<CUdeviceptr>(ptr_);
			count_ = 0;
			alloc_count_ = 0;
			ptr_ = nullptr;
			return current;
		}

		void upload(const T* data) {
			CUDA_CHECK(cudaMemcpy( ptr_, data, count_ * sizeof( T ), cudaMemcpyHostToDevice ));
		}

		void download(T* data) const {
			CUDA_CHECK(cudaMemcpy( data, ptr_, count_ * sizeof( T ), cudaMemcpyDeviceToHost ));
		}

		void downloadSub(size_t count, size_t offset, T* data) const {
			assert(count + offset < alloc_count_);
			CUDA_CHECK(cudaMemcpy( data, ptr_ + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ));
		}

		[[nodiscard]] size_t count() const { return count_; }
		[[nodiscard]] size_t reservedCount() const { return alloc_count_; }
		[[nodiscard]] size_t byteSize() const { return alloc_count_ * sizeof(T); }

	private:
		size_t count_ = 0;
		size_t alloc_count_ = 0;
		T* ptr_ = nullptr;
	};
}
