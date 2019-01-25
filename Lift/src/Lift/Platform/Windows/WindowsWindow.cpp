#include "pch.h"

#include "WindowsWindow.h"
#include <codecvt>

#include "Lift/Events/ApplicationEvent.h"
#include "Lift/Events/MouseEvent.h"
#include "Lift/Events/KeyEvent.h"


namespace Lift {
	
	static bool _sInitialized = false;

	Window* Window::Create(const WindowProps& props) {
		return new WindowsWindow(props);
	}

	LRESULT CALLBACK WindowsWindow::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {	
		WindowData* windowData;
		if(msg == WM_CREATE) {
			
			auto* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
			windowData = reinterpret_cast<WindowData*>(pCreate->lpCreateParams);
			SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(windowData));
		}
		else {
			windowData = GetWindowData(hwnd);
		}

		switch (msg) {
		
        case WM_CLOSE: {
            DestroyWindow(hwnd);
			WindowCloseEvent event;
			windowData->EventCallback(event);
            return 0;
		}
        case WM_DESTROY: {
			WindowCloseEvent event;
			windowData->EventCallback(event);
            PostQuitMessage(0);
            return 0;
		}
        case WM_KEYDOWN: {
	        KeyPressedEvent event(wParam, 0);
			windowData->EventCallback(event);
            if (wParam == VK_ESCAPE) PostQuitMessage(0);
            return 0;
        }
		case WM_KEYUP: {
	        KeyReleasedEvent event(wParam);
			windowData->EventCallback(event);
			return 0;
        }
		case WM_SIZE: {
			windowData->Width = LOWORD(lParam);
			windowData->Height = HIWORD(lParam);
			
			WindowResizeEvent event(windowData->Width, windowData->Height);
			if(windowData->EventCallback != nullptr)
				windowData->EventCallback(event);
			return 0;
		}
		case WM_LBUTTONDOWN: {
	        MouseButtonPressedEvent event(0);
			windowData->EventCallback(event);
			return 0;
        }
		case WM_MOUSEHWHEEL: {
			//TODO fix me 
	        MouseScrolledEvent event(0, static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)));
			windowData->EventCallback(event);
			return 0;
        }
		case  WM_MOUSEMOVE: {
			
	        MouseMovedEvent event(static_cast<float>(GET_X_LPARAM(lParam)), static_cast<float>(GET_Y_LPARAM(lParam)));
			windowData->EventCallback(event);
			return 0;
		}
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
        }
    }



	Lift::WindowsWindow::WindowsWindow(const WindowProps& props) {
		_data.Title = props.Title;
		_data.Width = props.Width;
		_data.Height = props.Height;
		
		LF_CORE_INFO("Creating Window {0} ({1} x {2})", props.Title, props.Width, props.Height);

		const WCHAR* className = L"LiftEngineWindowClass";
		const DWORD winStyle =  WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;

		WNDCLASS wc = {};
		wc.lpfnWndProc = MsgProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = className;	
		
		if (RegisterClass(&wc) == 0) {
            LF_CORE_ERROR("RegisterClass() failed");
            _windowHandle = nullptr;
        }

		RECT r {0, 0, static_cast<LONG>(props.Width), static_cast<LONG>(props.Height) };
		AdjustWindowRect(&r, winStyle, false);

		std::wstring wTitle = String2WString(props.Title);
		_windowHandle = CreateWindowEx(0, className, 
			wTitle.c_str(), winStyle,
			CW_USEDEFAULT, CW_USEDEFAULT,
			props.Width, props.Height, 
			nullptr, nullptr, 
			wc.hInstance, &_data);
		if(_windowHandle == nullptr) {
            LF_CORE_ERROR("CreateWindowEx() failed");
		}

		ShowWindow(_windowHandle, SW_SHOWNORMAL);
	}

	Lift::WindowsWindow::~WindowsWindow() {
		DestroyWindow(_windowHandle);
	}

	void Lift::WindowsWindow::OnUpdate() {
		MSG msg;
		if(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	unsigned Lift::WindowsWindow::GetWidth() const {
		return _data.Width;
	}

	unsigned Lift::WindowsWindow::GetHeight() const {
		return _data.Height;
	}

	void Lift::WindowsWindow::SetEventCallback(const EventCallbackFn& callback) {
		_data.EventCallback = callback;
	}

	void Lift::WindowsWindow::SetVSync(bool enabled) {
	}

	bool Lift::WindowsWindow::IsVSync() const {
		return true;
	}

	std::wstring WindowsWindow::String2WString(const std::string& s) {
		std::wstring_convert<std::codecvt_utf8<WCHAR>> cvt;
		std::wstring ws = cvt.from_bytes(s);
		return ws;
	}

	WindowsWindow::WindowData* WindowsWindow::GetWindowData(const HWND hwnd) {
		const LONG_PTR ptr = GetWindowLongPtr(hwnd, GWLP_USERDATA);
		auto* data = reinterpret_cast<WindowData*>(ptr);
		return data;
	}

}
