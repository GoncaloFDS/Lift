set(EXTERNAL_DIR ../external)
set(exe_name ${MAIN_PROJECT})

set(src_files_core
        core/glm.h
        core/stb_image_impl.cpp
        core/stb_image_impl.h
        events/application_event.h
        events/event.h
        events/key_event.h
        events/mouse_event.h
        )

set(src_files_assets
        assets/cornell_box.cpp
        assets/cornell_box.h
        assets/material.h
        assets/model.cpp
        assets/model.h
        assets/procedural.h
        assets/scene.cpp
        assets/scene.h
        assets/sphere.h
        assets/texture.cpp
        assets/texture.h
        assets/texture_image.cpp
        assets/texture_image.h
        assets/uniform_buffer.cpp
        assets/uniform_buffer.h
        assets/vertex.h
        )

set(src_files_imgui
        ImGui/imgui_impl_glfw.cpp
        ImGui/imgui_impl_glfw.h
        ImGui/imgui_impl_vulkan.cpp
        ImGui/imgui_impl_vulkan.h
        )

set(src_files_vulkan
        platform/vulkan/Application.cpp
        platform/vulkan/Application.h
        platform/vulkan/buffer.cpp
        platform/vulkan/buffer.h
        platform/vulkan/command_buffers.cpp
        platform/vulkan/command_buffers.h
        platform/vulkan/command_pool.cpp
        platform/vulkan/command_pool.h
        platform/vulkan/depth_buffer.cpp
        platform/vulkan/depth_buffer.h
        platform/vulkan/descriptor_binding.h
        platform/vulkan/descriptor_pool.cpp
        platform/vulkan/descriptor_pool.h
        platform/vulkan/descriptor_set_layout.cpp
        platform/vulkan/descriptor_set_layout.h
        platform/vulkan/descriptor_set_manager.cpp
        platform/vulkan/descriptor_set_manager.h
        platform/vulkan/descriptor_sets.cpp
        platform/vulkan/descriptor_sets.h
        platform/vulkan/device.cpp
        platform/vulkan/device.h
        platform/vulkan/device_memory.cpp
        platform/vulkan/device_memory.h
        platform/vulkan/enumerate.h
        platform/vulkan/fence.cpp
        platform/vulkan/fence.h
        platform/vulkan/frame_buffer.cpp
        platform/vulkan/frame_buffer.h
        platform/vulkan/graphics_pipeline.cpp
        platform/vulkan/graphics_pipeline.h
        platform/vulkan/image.cpp
        platform/vulkan/image.h
        platform/vulkan/image_memory_barrier.h
        platform/vulkan/image_view.cpp
        platform/vulkan/image_view.h
        platform/vulkan/instance.cpp
        platform/vulkan/instance.h
        platform/vulkan/pipeline_layout.cpp
        platform/vulkan/pipeline_layout.h
        platform/vulkan/render_pass.cpp
        platform/vulkan/render_pass.h
        platform/vulkan/sampler.cpp
        platform/vulkan/sampler.h
        platform/vulkan/semaphore.cpp
        platform/vulkan/semaphore.h
        platform/vulkan/shader_module.cpp
        platform/vulkan/shader_module.h
        platform/vulkan/single_time_commands.h
        platform/vulkan/surface.cpp
        platform/vulkan/surface.h
        platform/vulkan/swap_chain.cpp
        platform/vulkan/swap_chain.h
        core/utilities.cpp
        core/utilities.h
        platform/vulkan/window.cpp
        platform/vulkan/window.h
        platform/vulkan/window_properties.h
        )

set(src_files_vulkan_raytracing
        platform/vulkan/rayTracing/acceleration_structure.cpp
        platform/vulkan/rayTracing/acceleration_structure.h
        platform/vulkan/rayTracing/application.cpp
        platform/vulkan/rayTracing/application.h
        platform/vulkan/rayTracing/blas.cpp
        platform/vulkan/rayTracing/blas.h
        platform/vulkan/rayTracing/device_procedures.cpp
        platform/vulkan/rayTracing/device_procedures.h
        platform/vulkan/rayTracing/ray_tracing_pipeline.cpp
        platform/vulkan/rayTracing/ray_tracing_pipeline.h
        platform/vulkan/rayTracing/ray_tracing_properties.cpp
        platform/vulkan/rayTracing/ray_tracing_properties.h
        platform/vulkan/rayTracing/shader_binding_table.cpp
        platform/vulkan/rayTracing/shader_binding_table.h
        platform/vulkan/rayTracing/tlas.cpp
        platform/vulkan/rayTracing/tlas.h
        )

set(src_files
        main.cpp
        properties.cpp
        properties.h
        RayTracer.cpp
        RayTracer.h
        scene_list.cpp
        scene_list.h
        imgui_layer.cpp
        imgui_layer.h
        user_settings.h
        )

source_group("assets" FILES ${src_files_assets})
source_group("core" FILES ${src_files_core})
source_group("imGui" FILES ${src_files_imgui})
source_group("vulkan" FILES ${src_files_vulkan})
source_group("vulkan rayTracing" FILES ${src_files_vulkan_raytracing})
source_group("main" FILES ${src_files})

include_directories(.)
include_directories(./platform)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${glfw3_INCLUDE_DIRS})
include_directories(${glm_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${OptiX_INCLUDE})
include_directories(${EXTERNAL_DIR}/imgui)
include_directories(${EXTERNAL_DIR}/tinygltf)
include_directories(${EXTERNAL_DIR}/tinyobj)
include_directories(${EXTERNAL_DIR}/spdlog/include)
include_directories(${EXTERNAL_DIR}/glm)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${Vulkan_LIBRARY})

message(STATUS "Optix include dir: " ${OptiX_INCLUDE})

#ImGUI
add_library(imgui
        ${EXTERNAL_DIR}/imgui/imconfig.h
        ${EXTERNAL_DIR}/imgui/imgui.h
        ${EXTERNAL_DIR}/imgui/imgui.cpp
        ${EXTERNAL_DIR}/imgui/imgui_draw.cpp
        ${EXTERNAL_DIR}/imgui/imgui_internal.h
        ${EXTERNAL_DIR}/imgui/imgui_widgets.cpp
        ${EXTERNAL_DIR}/imgui/imstb_rectpack.h
        ${EXTERNAL_DIR}/imgui/imstb_textedit.h
        ${EXTERNAL_DIR}/imgui/imstb_truetype.h
        ${EXTERNAL_DIR}/imgui/imgui_demo.cpp)

add_executable(${exe_name}
        ${src_files_assets}
        ${src_files_imgui}
        ${src_files_core}
        ${src_files_vulkan}
        ${src_files_vulkan_raytracing}
        ${src_files}
        )

set_target_properties(${exe_name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_link_libraries(${exe_name} ${Boost_LIBRARIES} glfw imgui spdlog ${Vulkan_LIBRARIES} ${optix_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

add_dependencies(${exe_name} Assets)


