cmake_minimum_required(VERSION 3.14)

add_library(CudaPtxObjects OBJECT
        cuda/device_programs.cu
        cuda/local_geometry.h
        cuda/random.cuh
        )

set_property(TARGET CudaPtxObjects PROPERTY CUDA_PTX_COMPILATION ON)
set_target_properties(CudaPtxObjects
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/
        )
target_compile_definitions(CudaPtxObjects PUBLIC "PTX_FILES")
target_include_directories(CudaPtxObjects PRIVATE ../external/spdlog/include)
target_include_directories(CudaPtxObjects PRIVATE ../external/glm)
target_include_directories(CudaPtxObjects PRIVATE ./)

add_executable(Lift
        Sandbox.cpp
        core/io/Log.cpp
        core/io/Log.h
        core/io/tiny_gltf.h
        core/os/Input.h
        core/os/KeyCodes.h
        core/os/Window.h
        core/Layer.cpp
        core/Layer.h
        core/LayerStack.cpp
        core/LayerStack.h
        core/Profiler.cpp
        core/Profiler.h
        core/Timer.cpp
        core/Timer.h
        core/Util.cpp
        core/Util.h
        cuda/launch_parameters.h
        cuda/vec_math.h
        events/ApplicationEvent.h
        events/Event.h
        events/KeyEvent.h
        events/MouseEvent.h
        imgui/ImguiBuild.cpp
        imgui/ImguiLayer.cpp
        imgui/ImguiLayer.h
        platform/opengl/OpenGLBuffer.cpp
        platform/opengl/OpenGLBuffer.h
        platform/opengl/OpenGLContext.cpp
        platform/opengl/OpenGLContext.h
        platform/opengl/OpenGLRendererAPI.cpp
        platform/opengl/OpenGLRendererAPI.h
        platform/opengl/OpenGLVertexArray.cpp
        platform/opengl/OpenGLVertexArray.h
        platform/opengl/PixelBuffer.cpp
        platform/opengl/PixelBuffer.h
        platform/windows/WindowsInput.cpp
        platform/windows/WindowsInput.h
        platform/windows/WindowsWindow.cpp
        platform/windows/WindowsWindow.h
        renderer/Buffer.cpp
        renderer/Buffer.h
        renderer/BufferView.h
        renderer/CudaBuffer.h
        renderer/FrameBuffer.cpp
        renderer/FrameBuffer.h
        renderer/GraphicsContext.h
        renderer/Record.h
        renderer/RenderCommand.cpp
        renderer/RenderCommand.h
        renderer/Renderer.cpp
        renderer/Renderer.h
        renderer/RendererAPI.cpp
        renderer/RendererAPI.h
        renderer/Shader.cpp
        renderer/Shader.h
        renderer/Texture.cpp
        renderer/Texture.h
        renderer/VertexArray.cpp
        renderer/VertexArray.h
        scene/cameras/Camera.cpp
        scene/cameras/Camera.h
        scene/Aabb.cpp
        scene/Aabb.h
        scene/GeometryData.h
        scene/MaterialData.h
        scene/Mesh.h
        scene/Scene.cpp
        scene/Scene.h
        Application.cpp
        Application.h
        Core.h
        Lift.h
        Main.h
        pch.cpp
        pch.h
        scene/Light.h
        cuda/preprocessor.h
        cuda/math_constructors.h)


find_package(OptiX REQUIRED)
message("OptiX include dir: " ${OptiX_INCLUDE})
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
message("CUDA include dir: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
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

add_dependencies(Lift CudaPtxObjects)
add_custom_command(
        TARGET Lift PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/lift/res/ptx
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CudaPtxObjects> ${PROJECT_SOURCE_DIR}/lift/res/ptx/.
        COMMENT "Copying ptx file"
)
