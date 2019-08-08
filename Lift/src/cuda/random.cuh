#pragma once

#include <type_traits>
#include <crt/host_defines.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3

inline __device__ vec3 random_color(int i) {
	int r = unsigned(i)*13*17 + 0x234235;
    int g = unsigned(i)*7*3*5 + 0x773477;
    int b = unsigned(i)*11*19 + 0x223766;
    return vec3((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
}
