#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <utility>

#include <string>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>

#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtc/type_ptr.hpp> // value_ptr
#include <glm/ext.hpp> // glm::to_string
#include <glm/gtx/string_cast.hpp> // glm::to_string

using namespace glm;

#include "core/io/Log.h"
#include "core/os/KeyCodes.h"
#include "core/Util.h"

#ifdef LF_PLATFORM_WINDOWS
#include <Windows.h>
#include "Windowsx.h"
#endif

