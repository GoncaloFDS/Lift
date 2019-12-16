#pragma once
#include "cuda/launch_parameters.h"

namespace lift {
template<typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {
};

typedef Record<EmptyData> EmptyRecord;

typedef Record<RayGenData> RayGenRecord;
typedef Record<HitGroupData> HitGroupRecord;
typedef Record<MissData> MissDataRecord;

}
