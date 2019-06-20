project "Glad"
	kind "StaticLib"
	language "C"
	staticruntime "on"

	targetdir ("../../build/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("../../build/bin-int/" .. outputdir .. "/%{prj.name}")

	files {
		"include/glad/glad.h",
		"include/KHR/khrplatform.h",
		"src/glad.c"
	}

	includedirs {
		"include"
	}

	filter "system:windows"
		systemversion "latest"

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"