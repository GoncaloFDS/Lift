cmake_minimum_required(VERSION 3.14)
project(Lift LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(cmake/configure_build_type.cmake)
include(cmake/configure_optix.cmake)

mark_as_advanced(CUDA_SDK_ROO_DIR)

add_subdirectory(lift)
add_subdirectory(external/glfw)
add_subdirectory(external/spdlog)

target_link_directories(Lift PRIVATE lift/src)

