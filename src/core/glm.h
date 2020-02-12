#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE // vulkan [0, 1] depth range, instead of OpenGL [-1, +1]
#define GLM_FORCE_RIGHT_HANDED // vulkan has a left handed coordinate system (same as DirectX), OpenGL is right handed
#define GLM_FORCE_RADIANS
//#define GLM_FORCE_MESSAGES 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtx/quaternion.hpp> // glm::quaternion
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtc/type_ptr.hpp> // value_ptr
#include <glm/ext.hpp> // glm::to_string
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/hash.hpp>
