#pragma once

#include <optixu/optixu_math_namespace.h>                                        

struct ParallelogramLight                                                        
{                                                                                
    optix::float3 corner;                                                          
    optix::float3 v1, v2;                                                          
    optix::float3 normal;                                                          
    optix::float3 emission;                                                        
};