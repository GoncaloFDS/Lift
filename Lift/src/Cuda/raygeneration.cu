#include <optix.h>
#include <optixu/optixu_math_namespace.h>

// Note, the nomenclature used in the device code of all optixIntroduction samples
// follows some simple rules using prefixes to help indicating the scope and meaning:
//
// "sys" = renderer "system"-wide variables, defined at global context scope.
// "the" = variables with OptiX built-in semantic, like rtLaunchIndex, etc.
// "var" = "varyings" with developer defined attribute semantic, calculated by the intersection program.
// "par" = "parameter" variable held at some object scope, not at the global context scope.
//         (Exception to the last rule are the vertex "attributes" and "indices" held at Geometry nodes.)

rtBuffer<float4, 2> sysOutputBuffer; // RGBA32F

rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

rtDeclareVariable(float3, sysColorBackground, , );

// Entry point for simple color filling kernel.
RT_PROGRAM void raygeneration() {
  sysOutputBuffer[theLaunchIndex] = make_float4(sysColorBackground, 1.0f);
}
