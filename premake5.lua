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
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.glm}",
		"%{IncludeDir.ImGui}"
	}

	links {
		"GLFW",
		"Glad",
		"ImGui",
		"opengl32.lib" -- this might not be needed
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
		"%{prj.name}/src/**.cpp"
	}

	includedirs {
		"Lift/vendor/spdlog/include",
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