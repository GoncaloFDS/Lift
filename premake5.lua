workspace "Lift"
	architecture "x64"
	startproject "Sandbox"

	configurations {
		"Debug",
		"Release",
		"Dist"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- Set $(SolutionDir)tests\res\ptx\%(Filename).ptx  as CUDA compiler output
-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["spdlog"] = "external/spdlog"
IncludeDir["stbi"] = "external/stb_image"
IncludeDir["GLFW"] = "external/glfw/include"
IncludeDir["Glad"] = "external/glad/include"
IncludeDir["glm"] = "external/glm"
IncludeDir["ImGui"] = "external/imgui"
IncludeDir["optix"] = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
IncludeDir["cuda"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"

group "Dependencies"
	include "external/glfw"
	include "external/glad"
	include "external/imgui"
group ""

project "Lift"
	location "lift"
	kind "StaticLib"
	language "C++"
	cppdialect "c++17"
	staticruntime "on"

	targetdir ("build/bin/" .. outputdir)
	objdir ("build/bin-int/" .. outputdir)

	pchheader "pch.h"
	pchsource "lift/src/pch.cpp"

	files {
		"lift/src/**.h",
		"lift/src/**.cpp",
	}

	includedirs {
		"lift/src",
		"%{IncludeDir.spdlog}/include",
		"%{IncludeDir.stbi}",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.glm}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.optix}/include",
		"%{IncludeDir.cuda}/include"
	}

	links {
		"GLFW",
		"Glad",
		"ImGui",
		"%{IncludeDir.cuda}/lib/x64/cudart_static.lib",
		"%{IncludeDir.cuda}/lib/x64/cuda.lib",
	}
	
	postbuildcommands {
		--("{COPY} \"%{IncludeDir.optix}/bin64/*\" \"%{cfg.targetdir}\"")
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
	location "sandbox"
	kind "ConsoleApp"
	language "C++"
	cppdialect "c++17"
	staticruntime "on"

	targetdir ("build/bin/" .. outputdir)
	objdir ("build/bin-int/" .. outputdir)

	files {
		"sandbox/src/**.h",
		"sandbox/src/**.cpp"
	}

	includedirs {
		"lift/src",
		"external",
		"%{IncludeDir.spdlog}/include",
		"%{IncludeDir.glm}",
		"%{IncludeDir.optix}/include",
		"%{IncludeDir.cuda}/include"
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