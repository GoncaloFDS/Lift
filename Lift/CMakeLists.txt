cmake_minimum_required(VERSION 3.14)

add_library(CudaPtxObjects OBJECT
        src/cuda/device_programs.cu
        src/cuda/local_geometry.h
        src/cuda/random.cuh
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
target_include_directories(CudaPtxObjects PRIVATE src)

add_executable(Lift
        src/sandbox.cpp
        src/core/io/Log.cpp
        src/core/io/Log.h
        src/core/io/tiny_gltf.h
        src/core/os/Input.h
        src/core/os/KeyCodes.h
        src/core/os/Window.h
        src/core/Layer.cpp
        src/core/Layer.h
        src/core/LayerStack.cpp
        src/core/LayerStack.h
        src/core/Profiler.cpp
        src/core/Profiler.h
        src/core/Timer.cpp
        src/core/Timer.h
        src/core/Util.cpp
        src/core/Util.h
        src/cuda/launch_parameters.h
        src/cuda/vec_math.h
        src/events/ApplicationEvent.h
        src/events/Event.h
        src/events/KeyEvent.h
        src/events/MouseEvent.h
        src/imgui/ImguiBuild.cpp
        src/imgui/ImguiLayer.cpp
        src/imgui/ImguiLayer.h
        src/platform/opengl/OpenGLBuffer.cpp
        src/platform/opengl/OpenGLBuffer.h
        src/platform/opengl/OpenGLContext.cpp
        src/platform/opengl/OpenGLContext.h
        src/platform/opengl/OpenGLRendererAPI.cpp
        src/platform/opengl/OpenGLRendererAPI.h
        src/platform/opengl/OpenGLVertexArray.cpp
        src/platform/opengl/OpenGLVertexArray.h
        src/platform/opengl/PixelBuffer.cpp
        src/platform/opengl/PixelBuffer.h
        src/platform/windows/WindowsInput.cpp
        src/platform/windows/WindowsInput.h
        src/platform/windows/WindowsWindow.cpp
        src/platform/windows/WindowsWindow.h
        src/renderer/Buffer.cpp
        src/renderer/Buffer.h
        src/renderer/BufferView.h
        src/renderer/CudaBuffer.h
        src/renderer/FrameBuffer.cpp
        src/renderer/FrameBuffer.h
        src/renderer/GraphicsContext.h
        src/renderer/Record.h
        src/renderer/RenderCommand.cpp
        src/renderer/RenderCommand.h
        src/renderer/Renderer.cpp
        src/renderer/Renderer.h
        src/renderer/RendererAPI.cpp
        src/renderer/RendererAPI.h
        src/renderer/Shader.cpp
        src/renderer/Shader.h
        src/renderer/Texture.cpp
        src/renderer/Texture.h
        src/renderer/VertexArray.cpp
        src/renderer/VertexArray.h
        src/scene/cameras/Camera.cpp
        src/scene/cameras/Camera.h
        src/scene/Aabb.cpp
        src/scene/Aabb.h
        src/scene/GeometryData.h
        src/scene/MaterialData.h
        src/scene/Mesh.h
        src/scene/Scene.cpp
        src/scene/Scene.h
        src/Application.cpp
        src/Application.h
        src/Core.h
        src/Lift.h
        src/Main.h
        src/pch.cpp
        src/pch.h
        src/scene/Light.h src/cuda/preprocessor.h src/cuda/math_constructors.h)

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
        TARGET Lift
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/lift/res/ptx
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:CudaPtxObjects> ${PROJECT_SOURCE_DIR}/lift/res/ptx/.
        COMMENT "Copying ptx file"
)
