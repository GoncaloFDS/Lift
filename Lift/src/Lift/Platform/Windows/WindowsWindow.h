#pragma once

#include "Lift/Window.h"
#include "GLFW/glfw3.h"

namespace Lift {

	class LIFT_API WindowsWindow : public Window {
	public:
		WindowsWindow(const WindowProps& props);
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
		virtual void Init(const WindowProps& props);
		virtual void Shutdown();

	private:
		GLFWwindow* m_windowHandle{};

		struct WindowData {
			std::string title;
			unsigned int width{}, height{};
			bool vSync{};

			EventCallbackFn eventCallback;

		};

		WindowData m_data;
	};

}
