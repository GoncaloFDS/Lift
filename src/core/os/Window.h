#pragma once

#include "pch.h"

#include "events/Event.h"
#include "Renderer/GraphicsContext.h"

namespace lift {

	struct WindowProperties {
		std::string title;
		unsigned int width;
		unsigned int height;
		unsigned int x, y;

		WindowProperties(std::string title = "lift Engine",
						 const unsigned int width = 1280,
						 const unsigned int height = 720,
						 const unsigned int position_x = 200,
						 const unsigned int position_y = 200)
			: title(std::move(title)), width(width), height(height), x(position_x), y(position_y) {
		}

	};

	// Window Interface to be implemented for each platform
	class Window {
	public:
		using EventCallbackFn = std::function<void(Event&)>;

		virtual ~Window() = default;

		virtual void OnUpdate() = 0;

		[[nodiscard]] virtual unsigned int Width() const = 0;
		[[nodiscard]] virtual unsigned int Height() const = 0;
		[[nodiscard]] virtual std::pair<int, int> GetPosition() const = 0;

		// Window attributes
		virtual void SetEventCallback(const EventCallbackFn& callback) = 0;
		virtual void SetVSync(bool enabled) = 0;
		virtual bool IsVSync() const = 0;

		virtual void* GetNativeWindow() const = 0;

		static Window* Create(const WindowProperties& props = WindowProperties());

	};
}
