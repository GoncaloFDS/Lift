cmake_minimum_required(VERSION 3.10)
project(Lift LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(cmake/configure_build_type.cmake)
include(cmake/configure_optix.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

add_subdirectory(src)
add_subdirectory(external/glfw)
add_subdirectory(external/spdlog)

target_link_directories(Lift PRIVATE lift/src)

message("Copying resources folder")
file(COPY ${PROJECT_SOURCE_DIR}/res/ DESTINATION ${PROJECT_BINARY_DIR}/res/)
