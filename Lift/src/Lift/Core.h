#pragma once

#ifdef LF_PLATFORM_WINDOWS
	#ifdef LF_BUILD_DLL
		#define LIFT_API __declspec(dllexport)
	#else
		#define LIFT_API __declspec(dllimport)
	#endif
#else
	#error Lift only supports windows
#endif

#define BIT(x) (1 << x)