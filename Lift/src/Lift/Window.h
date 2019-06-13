#pragma once

#include "pch.h"

#include "Lift/Events/Event.h"

namespace lift {

	struct WindowProperties {
		std::string title;
		unsigned int width;
		unsigned int height;

		WindowProperties(std::string title = "lift Engine",
		            const unsigned int width = 1280,
		            const unsigned int height = 720)
			: title(std::move(title)), width(width), height(height) {
		}

	};

	// Window Interface to be implemented for each platform
	class Window {
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

		virtual void* GetNativeWindow() const = 0;

		static Window* Create(const WindowProperties& props = WindowProperties());
	};
}
