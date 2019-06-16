workspace "Lift"
	architecture "x64"
	startproject "Sandbox"

	configurations {
		"Debug",
		"Release",
		"Dist"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["ImGui"] = "Lift/vendor/imgui"
IncludeDir["GLFW"] = "Lift/vendor/glfw/include"
IncludeDir["Glad"] = "Lift/vendor/glad/include"
IncludeDir["glm"] = "Lift/vendor/glm"
IncludeDir["mathfu"] = "Lift/vendor/mathfu/Include"
IncludeDir["optix"] = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.0.0"
IncludeDir["cuda"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"

group "Dependencies"
	include "Lift/vendor/glfw"
	include "Lift/vendor/glad"
	include "Lift/vendor/imgui"
group ""

project "Lift"
	location "Lift"
	kind "StaticLib"
	language "C++"
	cppdialect "c++17"
	staticruntime "on"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "pch.h"
	pchsource "Lift/src/pch.cpp"

	files {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}

	includedirs {
		"%{prj.name}/src",
		"%{prj.name}/vendor/spdlog/include",
		"%{prj.name}/vendor/mathfu/include",
		"%{prj.name}/vendor/stb_image",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.optix}/include",
		--"%{IncludeDir.optix}/SDK",
		--"%{IncludeDir.optix}/SDK/sutil",
		"%{IncludeDir.optix}/include/optixu",
		--"%{IncludeDir.optix}/SDK/build",
		"%{IncludeDir.cuda}/include"
	}

	links {
		"GLFW",
		"Glad",
		"ImGui",
		"opengl32.lib",
		"%{IncludeDir.cuda}/lib/x64/nvrtc.lib",
		"%{IncludeDir.optix}/lib64/optix.6.0.0.lib",
	}

	defines {
		"_CRT_SECURE_NO_WARNINGS"
	}

	filter "system:windows"
		systemversion "latest"

		defines {
			"LF_PLATFORM_WINDOWS",
			"LF_BUILD_DLL",
			"GLFW_INCLUDE_NONE"
		}

	filter "configurations:Debug"
		defines "LF_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		defines "LF_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Dist"
		defines "LF_DIST"
		runtime "Release"
		optimize "on"

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"
	cppdialect "c++17"
	staticruntime "on"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/res/**"
	}

	includedirs {
		"Lift/vendor/spdlog/include",
		"Lift/vendor/mathfu/include",
		"Lift/src",
		"Lift/vendor",
		"%{IncludeDir.glm}"
	}

	links {
		"Lift"
	}

	filter "system:windows"
		systemversion "latest"

		defines {
			"LF_PLATFORM_WINDOWS"
		}

	filter "configurations:Debug"
		defines "LF_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		defines "LF_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Dist"
		defines "LF_DIST"
		runtime "Release"
		optimize "on"