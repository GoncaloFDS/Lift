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

include "Lift/vendor/glfw"
include "Lift/vendor/glad"
include "Lift/vendor/imgui"

project "Lift"
	location "Lift"
	kind "SharedLib"
	language "C++"
	staticruntime "off"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("intermediate/" .. outputdir .. "/%{prj.name}")

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
		"%{IncludeDir.ImGui}"
	}

	links {
		"GLFW",
		"Glad",
		"opengl32.lib", -- this might not be needed
		"ImGui",
		"d3dcompiler", 
		"dxguid", 
		"d3d12", 
		"dxgi"
	}

	filter "system:windows"
		cppdialect "c++17"
		systemversion "latest"

		defines {
			"LF_PLATFORM_WINDOWS",
			"LF_BUILD_DLL",
			"GLFW_INCLUDE_NONE"
		}

		postbuildcommands {
			("{COPY} %{cfg.buildtarget.relpath} ../bin/" .. outputdir .. "/Sandbox")
		}

	filter "configurations:Debug"
		defines{
			"LF_DEBUG",
			"LF_ENABLE_ASSERTS", 
			"Win32"
		} 
		buildoptions "/MDd"
		symbols "On"

	filter "configurations:Release"
		defines "LF_RELEASE"
		defines "LF_ENABLE_ASSERTS"
		buildoptions "/MD"
		optimize "On"

	filter "configurations:Dist"
		defines "LF_DIST"
		buildoptions "/MD"
		optimize "On"

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"
	staticruntime "off"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("intermediate/" .. outputdir .. "/%{prj.name}")

	files {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}

	includedirs {
		"Lift/vendor/spdlog/include",
		"Lift/src",
		"Lift/vendor"
	}

	links {
		"Lift"
	}

	filter "system:windows"
		cppdialect "c++17"
		systemversion "latest"

		defines {
			"LF_PLATFORM_WINDOWS"
		}

	filter "configurations:Debug"
		defines "LF_DEBUG"
		buildoptions "/MDd"
		symbols "On"

	filter "configurations:Release"
		defines "LF_RELEASE"
		buildoptions "/MD"
		optimize "On"

	filter "configurations:Dist"
		defines "LF_DIST"
		buildoptions "/MD"
		optimize "On"