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

#define print_x 512
#define print_y 384

#define print_pixel(...)                                                       \
{                                                                              \
    const uint3  idx__ = optixGetLaunchIndex();                                \
    if( idx__.x == print_y && idx__.y == print_x )                             \
        printf( __VA_ARGS__ );                                                 \
}
