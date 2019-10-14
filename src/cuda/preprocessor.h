#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define HOSTDEVICE __host__ __device__
#    define INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define HOSTDEVICE
#    define INLINE inline
#    define CONST_STATIC_INIT(...) = __VA_ARGS__
#endif
