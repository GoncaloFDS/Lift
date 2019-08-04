#include <optix.h>
#include <OptiX_world.h>

#include "ray_payload.cuh"

rtDeclareVariable(ShadowRayPayload, the_shadow_payload, rtPayload, );

RT_PROGRAM void any_hit_shadow() {
	the_shadow_payload.visible = false;
	rtTerminateRay();
}