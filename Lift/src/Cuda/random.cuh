#include <optixu/optixu_math_namespace.h>

template <unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(const unsigned int val0, const unsigned int val1) {
	auto v0 = val0;
	auto v1 = val1;
	unsigned int s0 = 0;

	for (unsigned int n = 0; n < N; n++) {
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev) {
	const auto lcg_a = 1664525u;
	const auto lcg_c = 1013904223u;
	prev = (lcg_a * prev + lcg_c);
	return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int& prev) {
	prev = (prev * 8121 + 28411) % 134456;
	return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int& prev) {
	return (static_cast<float>(lcg(prev)) / static_cast<float>(0x01000000));
}

static __host__ __device__ __inline__ unsigned int rot_seed(const unsigned int seed, const unsigned int frame) {
	return seed ^ frame;
}
