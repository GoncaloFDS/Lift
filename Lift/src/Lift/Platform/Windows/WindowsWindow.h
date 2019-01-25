#pragma once

#include "Lift/Window.h"

namespace Lift {
	
	class WindowsWindow : public Window {
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

		void OnSize(HWND hwnd, UINT flag, int width, int height);
		static LRESULT MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
	private:

		static std::wstring String2WString(const std::string& s);
	private:
		HWND _windowHandle;

		struct WindowData {
			std::string Title;
			unsigned int Width{}, Height{};
			bool VSync{};

			EventCallbackFn EventCallback;

		};

		WindowData _data;
		inline static WindowsWindow::WindowData* GetWindowData(HWND hwnd);
	};

}
