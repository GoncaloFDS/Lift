#pragma once

#include "Lift/Window.h"
#include "GLFW/glfw3.h"

namespace lift {

	class WindowsWindow : public Window {
	public:
		WindowsWindow(const WindowProperties& props);
		virtual ~WindowsWindow();

		void OnUpdate() override;

		inline unsigned int GetWidth() const override;
		inline unsigned int GetHeight() const override;

		//Window attributes
		inline void SetEventCallback(const EventCallbackFn& callback) override;
		void SetVSync(bool enabled) override;
		bool IsVSync() const override;

		inline void* GetNativeWindow() const override;
	private:
		virtual void Init(const WindowProperties& props);
		virtual void Shutdown();

	private:
		GLFWwindow* window_handle_{};

		struct WindowData {
			std::string title;
			unsigned int width{}, height{};
			bool v_sync{};

			EventCallbackFn event_callback;

		};

		WindowData properties_;
	};

}
