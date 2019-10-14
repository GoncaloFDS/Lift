cmake_minimum_required(VERSION 3.14)

cuda_compile_and_embed(embedded_ptx_code cuda/device_programs.cu)

add_executable(Lift
        ${embedded_ptx_code}
        sandbox.cpp
        core/io/log.cpp
        core/io/log.h
        core/io/tiny_gltf.h
        core/os/input.h
        core/os/key_codes.h
        core/os/window.h
        core/layer.cpp
        core/layer.h
        core/layer_stack.cpp
        core/layer_stack.h
        core/profiler.cpp
        core/profiler.h
        core/timer.cpp
        core/timer.h
        core/util.cpp
        core/util.h
        cuda/launch_parameters.h
        cuda/vec_math.h
        events/application_event.h
        events/event.h
        events/key_event.h
        events/mouse_event.h
        imgui/imgui_build.cpp
        imgui/imgui_layer.cpp
        imgui/imgui_layer.h
        platform/opengl/opengl_buffer.cpp
        platform/opengl/opengl_buffer.h
        platform/opengl/opengl_context.cpp
        platform/opengl/opengl_context.h
        platform/opengl/opengl_renderer_api.cpp
        platform/opengl/opengl_renderer_api.h
        platform/opengl/opengl_vertex_array.cpp
        platform/opengl/opengl_vertex_array.h
        platform/opengl/pixel_buffer.cpp
        platform/opengl/pixel_buffer.h
        platform/windows/windows_input.cpp
        platform/windows/windows_input.h
        platform/windows/windows_window.cpp
        platform/windows/windows_window.h
        renderer/buffer.cpp
        renderer/buffer.h
        cuda/buffer_view.h
        renderer/cuda_output_buffer.h
        renderer/graphics_context.h
        renderer/record.h
        renderer/render_command.cpp
        renderer/render_command.h
        renderer/renderer.cpp
        renderer/renderer.h
        renderer/renderer_api.cpp
        renderer/renderer_api.h
        renderer/shader.cpp
        renderer/shader.h
        renderer/texture.cpp
        renderer/texture.h
        renderer/vertex_array.cpp
        renderer/vertex_array.h
        scene/cameras/camera.cpp
        scene/cameras/camera.h
        scene/aabb.cpp
        scene/aabb.h
        cuda/geometry_data.h
        cuda/material_data.h
        scene/mesh.h
        scene/scene.cpp
        scene/scene.h
        application.cpp
        application.h
        core.h
        lift.h
        main.h
        pch.cpp
        pch.h
        cuda/light.h
        cuda/preprocessor.h
        cuda/math_constructors.h platform/opengl/opengl_display.cpp platform/opengl/opengl_display.h renderer/cuda_buffer.h)


find_package(OptiX REQUIRED)
target_include_directories(Lift PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ../external/glm
        ../external/glad/include
        ${OptiX_INCLUDE}
        )
target_include_directories(Lift PRIVATE
        src/
        ../external/tinygltf)

target_link_libraries(Lift
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARIES})

#Cuda
target_include_directories(Lift PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#GLAD
add_library(glad ../external/glad/src/glad.c)
target_include_directories(glad PRIVATE ../external/glad/include)
target_link_libraries(Lift glad ${CMAKE_DL_LIBS})

#GLFW
target_link_libraries(Lift glfw)

#ImGUI
add_library(imgui
        ../external/imgui/imconfig.h
        ../external/imgui/imgui.h
        ../external/imgui/imgui.cpp
        ../external/imgui/imgui_draw.cpp
        ../external/imgui/imgui_internal.h
        ../external/imgui/imgui_widgets.cpp
        ../external/imgui/imstb_rectpack.h
        ../external/imgui/imstb_textedit.h
        ../external/imgui/imstb_truetype.h
        ../external/imgui/imgui_demo.cpp
        )
target_include_directories(Lift PRIVATE ../external/imgui)
target_link_libraries(Lift imgui)

#spdlog
target_include_directories(Lift PUBLIC ../external/spdlog/include)
target_link_libraries(Lift spdlog)

add_compile_definitions(LF_PLATFORM_WINDOWS)
add_compile_definitions(LF_BUILD_DLL)
#add_compile_definitions(GLFW_INCLUDE_NONE)
add_compile_definitions(LF_DEBUG)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

