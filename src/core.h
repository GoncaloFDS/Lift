#pragma once

#ifdef LF_DEBUG
#define LF_ENABLE_ASSERTS
#endif

#ifdef LF_ENABLE_ASSERTS
#define LF_ASSERT(x, ...) { if(!(x)) { LF_ERROR(__VA_ARGS__); __debugbreak(); }}
#else
#define LF_ASSERT(x, ...)
#define LF_CORE_ASSERT(x, ...)
#endif

//#define BIT(x) (1 << (x))
template<typename T>
constexpr T bit(T x) {
    return 1 << x;
}

#define LF_BIND_EVENT_FN(fn) std::bind(&fn, this, std::placeholders::_1)

#ifdef LF_DEBUG
#define GL_CHECK(x) \
        (x); \
        while (GLenum error = glGetError()) { \
            LF_ERROR("[OpenGL Error] {0}", error);    \
            LF_ERROR("\tFile: {0}", __FILE__);    \
            LF_ERROR("\tLine: {0}", __LINE__);    \
        }
#else
#define GL_CHECK(x) (x);
#endif

#ifdef LF_DEBUG
#define OPTIX_CHECK(call) {                                                                   \
    OptixResult res = call;                                                                     \
    if( res != OPTIX_SUCCESS ) {                                                                \
        LF_FATAL("Optix call {0} failed with code {1} (line {2})", #call, res, __LINE__ ); \
        exit( 2 );                                                                              \
    }                                                                                            \
}
#else
#define OPTIX_CHECK( call ) ( call );
#endif

#define CUDA_CHECK(call)                                                            \
    do                                                                                \
    {                                                                                \
        cudaError_t error = call;                                                    \
        if( error != cudaSuccess )                                                    \
        {                                                                            \
            LF_FATAL("CUDA call {0} failed with code {1} (file {2} line {3})",    \
                            #call, cudaGetErrorString(error), __FILE__, __LINE__);  \
        }                                                                            \
    } while( 0 )

#define CUDA_CHECK_NOEXCEPT(call)                                        \
    {                                    \
      cuda##call;                                                       \
    }

#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if( error != cudaSuccess )                                          \
      {                                                                 \
        LF_ERROR("error ({0}: line {1}): {2}", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit( 2 );                                                      \
      }                                                                 \
  }

#define LF_FORWARD glm::vec3(0.0f, 0.0f, -1.0f)
#define LF_RIGHT glm::vec3(1.0f, 0.0f, 0.0f)
#define LF_UP glm::vec3(0.0f, 1.0f, 0.0f)
