#include <optix.h>

rtBuffer<float4, 2> sys_output_buffer; // RGBA32F

rtDeclareVariable(uint2, the_launch_index, rtLaunchIndex, );

RT_PROGRAM void exception() {
	const unsigned int code = rtGetExceptionCode();
	if (RT_EXCEPTION_USER <= code) {
		rtPrintf("User exception %d at (%d, %d)\n", code - RT_EXCEPTION_USER, the_launch_index.x, the_launch_index.y);
	}
	else {
		rtPrintf("Exception code 0x%X at (%d, %d)\n", code, the_launch_index.x, the_launch_index.y);
	}
	// RGBA32F super magenta as error color (makes sure this isn't accumulated away in a progressive renderer).
	sys_output_buffer[the_launch_index] = make_float4(1000000.0f, 0.0f, 1000000.0f, 1.0f);
}
