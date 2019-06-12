#pragma once

#ifdef LF_DEBUG
	#define LF_ENABLE_ASSERTS
#endif

#ifdef LF_ENABLE_ASSERTS
	#define LF_ASSERT(x, ...) { if(!(x)) { LF_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); }}
	#define LF_CORE_ASSERT(x, ...) { if(!(x)) { LF_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); }}
#else 
	#define LF_ASSERT(x, ...)
	#define LF_CORE_ASSERT(x, ...)
#endif

//#define BIT(x) (1 << (x))
template<typename T>
constexpr T Bit(T x) {
	return 1 << x;
}

#define LF_BIND_EVENT_FN(fn) std::bind(&fn, this, std::placeholders::_1)
