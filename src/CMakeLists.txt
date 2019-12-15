set(EXTERNAL_DIR ../external)
set(exe_name ${MAIN_PROJECT})

set(src_files_assets
        assets/CornellBox.cpp
        assets/CornellBox.hpp
        assets/Material.hpp
        assets/Model.cpp
        assets/Model.hpp
        assets/Procedural.hpp
        assets/Scene.cpp
        assets/Scene.hpp
        assets/Sphere.hpp
        assets/Texture.cpp
        assets/Texture.hpp
        assets/TextureImage.cpp
        assets/TextureImage.hpp
        assets/UniformBuffer.cpp
        assets/UniformBuffer.hpp
        assets/Vertex.hpp
        )

set(src_files_imgui
        ImGui/imgui_impl_glfw.cpp
        ImGui/imgui_impl_glfw.h
        ImGui/imgui_impl_vulkan.cpp
        ImGui/imgui_impl_vulkan.h
        )

set(src_files_utilities
        Utilities/Console.cpp
        Utilities/Console.hpp
        Utilities/Glm.hpp
        Utilities/StbImage.cpp
        Utilities/StbImage.hpp
        )

set(src_files_vulkan
        platform/vulkan/Application.cpp
        platform/vulkan/Application.h
        platform/vulkan/Buffer.cpp
        platform/vulkan/Buffer.h
        platform/vulkan/CommandBuffers.cpp
        platform/vulkan/CommandBuffers.h
        platform/vulkan/CommandPool.cpp
        platform/vulkan/CommandPool.h
        platform/vulkan/DepthBuffer.cpp
        platform/vulkan/DepthBuffer.h
        platform/vulkan/DescriptorBinding.h
        platform/vulkan/DescriptorPool.cpp
        platform/vulkan/DescriptorPool.h
        platform/vulkan/DescriptorSetLayout.cpp
        platform/vulkan/DescriptorSetLayout.h
        platform/vulkan/DescriptorSetManager.cpp
        platform/vulkan/DescriptorSetManager.h
        platform/vulkan/DescriptorSets.cpp
        platform/vulkan/DescriptorSets.h
        platform/vulkan/Device.cpp
        platform/vulkan/Device.h
        platform/vulkan/DeviceMemory.cpp
        platform/vulkan/DeviceMemory.h
        platform/vulkan/DebugUtilsMessenger.cpp
        platform/vulkan/DebugUtilsMessenger.h
        platform/vulkan/Enumerate.h
        platform/vulkan/Fence.cpp
        platform/vulkan/Fence.h
        platform/vulkan/FrameBuffer.cpp
        platform/vulkan/FrameBuffer.h
        platform/vulkan/GraphicsPipeline.cpp
        platform/vulkan/GraphicsPipeline.h
        platform/vulkan/Image.cpp
        platform/vulkan/Image.h
        platform/vulkan/ImageMemoryBarrier.h
        platform/vulkan/ImageView.cpp
        platform/vulkan/ImageView.h
        platform/vulkan/Instance.cpp
        platform/vulkan/Instance.h
        platform/vulkan/PipelineLayout.cpp
        platform/vulkan/PipelineLayout.h
        platform/vulkan/RenderPass.cpp
        platform/vulkan/RenderPass.h
        platform/vulkan/Sampler.cpp
        platform/vulkan/Sampler.h
        platform/vulkan/Semaphore.cpp
        platform/vulkan/Semaphore.h
        platform/vulkan/ShaderModule.cpp
        platform/vulkan/ShaderModule.h
        platform/vulkan/SingleTimeCommands.h
        platform/vulkan/Strings.cpp
        platform/vulkan/Strings.hpp
        platform/vulkan/Surface.cpp
        platform/vulkan/Surface.h
        platform/vulkan/SwapChain.cpp
        platform/vulkan/SwapChain.h
        platform/vulkan/Version.hpp
        platform/vulkan/VulkanError.cpp
        platform/vulkan/VulkanError.h
        platform/vulkan/Window.cpp
        platform/vulkan/Window.h
        platform/vulkan/WindowConfig.h
        )

set(src_files_vulkan_raytracing
        platform/vulkan/rayTracing/AccelerationStructure.cpp
        platform/vulkan/rayTracing/AccelerationStructure.hpp
        platform/vulkan/rayTracing/Application.cpp
        platform/vulkan/rayTracing/Application.hpp
        platform/vulkan/rayTracing/BottomLevelAccelerationStructure.cpp
        platform/vulkan/rayTracing/BottomLevelAccelerationStructure.hpp
        platform/vulkan/rayTracing/DeviceProcedures.cpp
        platform/vulkan/rayTracing/DeviceProcedures.hpp
        platform/vulkan/rayTracing/RayTracingPipeline.cpp
        platform/vulkan/rayTracing/RayTracingPipeline.hpp
        platform/vulkan/rayTracing/RayTracingProperties.cpp
        platform/vulkan/rayTracing/RayTracingProperties.hpp
        platform/vulkan/rayTracing/ShaderBindingTable.cpp
        platform/vulkan/rayTracing/ShaderBindingTable.hpp
        platform/vulkan/rayTracing/TopLevelAccelerationStructure.cpp
        platform/vulkan/rayTracing/TopLevelAccelerationStructure.hpp
        )

set(src_files
        main.cpp
        Options.cpp
        Options.h
        RayTracer.cpp
        RayTracer.h
        SceneList.cpp
        SceneList.h
        UserInterface.cpp
        UserInterface.h
        UserSettings.h
        )

source_group("Assets" FILES ${src_files_assets})
source_group("ImGui" FILES ${src_files_imgui})
source_group("Utilities" FILES ${src_files_utilities})
source_group("Vulkan" FILES ${src_files_vulkan})
source_group("Vulkan.RayTracing" FILES ${src_files_vulkan_raytracing})
source_group("Main" FILES ${src_files})

include_directories(.)
include_directories(./platform)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${glfw3_INCLUDE_DIRS})
include_directories(${glm_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})
link_directories(${Vulkan_LIBRARY})

find_package(OptiX REQUIRED)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE})

include_directories(
        src/
        ${EXTERNAL_DIR}/tinygltf
        ${EXTERNAL_DIR}/tinyobj)

link_libraries(
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARIES})

#Cuda
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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
include_directories(${EXTERNAL_DIR}/imgui)

#spdlog
#include_directories(${EXTERNAL_DIR}/spdlog/include)

#glm
include_directories(${EXTERNAL_DIR}/glm)

#boost

add_executable(${exe_name}
        ${src_files_assets}
        ${src_files_imgui}
        ${src_files_utilities}
        ${src_files_vulkan}
        ${src_files_vulkan_raytracing}
        ${src_files}
        )

#add_compile_definitions(LF_DEBUG)
#add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
set_target_properties(${exe_name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_link_libraries(${exe_name} PRIVATE ${Boost_LIBRARIES} glfw imgui spdlog ${Vulkan_LIBRARIES})
add_dependencies(${exe_name} Assets)


