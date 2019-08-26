#pragma once
#include <optix.h>
#include <crt/host_defines.h>
#include "cuda/launch_parameters.cuh"

namespace lift {
	template <typename T>
	struct Record {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	struct EmptyData {
	};

	typedef Record<EmptyData> EmptyRecord;

	typedef Record<HitGroupData> HitGroupRecord;
}
