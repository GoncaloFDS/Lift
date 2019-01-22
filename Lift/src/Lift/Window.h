#pragma once

#include "pch.h"

#include "Lift/Core.h"
#include "Lift/Events/Event.h"

namespace Lift {
	
	struct WindowProps {
		std::string Title;
		unsigned int Width;
		unsigned int Height;

		WindowProps(std::string title = "Lift Engine",
		            const unsigned int width = 1280,
		            const unsigned int height = 720 )
			: Title(std::move(title)), Width(width), Height(height)
		{}
			
	};

	// Window Interface to be implemented for each platform
	class LIFT_API Window {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() = default;

		virtual void OnUpdate() = 0;

		virtual unsigned int GetWidth() const = 0;
		virtual unsigned int GetHeight() const = 0;

		// Window attributes
		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		static Window* Create(const WindowProps& props = WindowProps());
	};
}