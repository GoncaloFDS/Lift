cmake_minimum_required(VERSION 3.14)

set(EXTERNAL_DIR ../external)

cuda_compile_and_embed(embedded_ptx_code cuda/path_tracer.cu)

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
        platform/opengl/opengl_context.cpp
        platform/opengl/opengl_context.h
        platform/windows/windows_input.cpp
        platform/windows/windows_input.h
        platform/windows/windows_window.cpp
        platform/windows/windows_window.h
        cuda/buffer_view.h
        renderer/cuda_output_buffer.h
        renderer/graphics_context.h
        renderer/record.h
        renderer/renderer.cpp
        renderer/renderer.h
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
        cuda/math_constructors.h
        renderer/cuda_buffer.h
        imgui/ui_elements.h
        core/os/input.cpp
        renderer/optix_context.cpp
        renderer/optix_context.h
        cuda/gdt.h)



find_package(OptiX REQUIRED)
target_include_directories(Lift PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        )

target_include_directories(Lift PRIVATE
        src/
        ${EXTERNAL_DIR}/tinygltf)

target_link_libraries(Lift
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARIES})

#Cuda
target_include_directories(Lift PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#GLAD
add_library(glad ${EXTERNAL_DIR}/glad/src/glad.c)
target_include_directories(glad PUBLIC ${EXTERNAL_DIR}/glad/include)
target_link_libraries(Lift glad ${CMAKE_DL_LIBS})

#GLFW
target_link_libraries(Lift glfw)

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
target_include_directories(Lift PRIVATE ${EXTERNAL_DIR}/imgui)
target_link_libraries(Lift imgui)

#spdlog
target_include_directories(Lift PUBLIC ${EXTERNAL_DIR}/spdlog/include)
target_link_libraries(Lift spdlog)

#glm
target_include_directories(Lift PUBLIC ${EXTERNAL_DIR}/glm)

add_compile_definitions(LF_PLATFORM_WINDOWS)
add_compile_definitions(LF_BUILD_DLL)
#add_compile_definitions(GLFW_INCLUDE_NONE)
add_compile_definitions(LF_DEBUG)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

