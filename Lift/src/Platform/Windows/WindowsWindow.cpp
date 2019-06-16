#include "pch.h"

#include "WindowsWindow.h"

#include "Events/ApplicationEvent.h"
#include "Events/MouseEvent.h"
#include "Events/KeyEvent.h"


namespace lift {

	static bool glfw_initialized = false;

	static void GLFWErrorCallback(const int error, const char* description) {
		LF_CORE_ERROR("GLFW Error ({0}): {1}", error, description);

	}

	Window* Window::Create(const WindowProperties& props) {
		return new WindowsWindow(props);
	}

	WindowsWindow::WindowsWindow(const WindowProperties& props) {
		WindowsWindow::Init(props);
	}

	WindowsWindow::~WindowsWindow() {
		WindowsWindow::Shutdown();
	}

	void WindowsWindow::OnUpdate() {
		glfwPollEvents();
	}

	unsigned WindowsWindow::GetWidth() const {
		return properties_.width;
	}

	unsigned WindowsWindow::GetHeight() const {
		return properties_.height;
	}

	void WindowsWindow::SetEventCallback(const EventCallbackFn& callback) {
		properties_.event_callback = callback;
	}

	void WindowsWindow::SetVSync(const bool enabled) {
		if (enabled)
			glfwSwapInterval(1);
		else
			glfwSwapInterval(0);
		properties_.v_sync = enabled;
	}

	bool WindowsWindow::IsVSync() const {
		return properties_.v_sync;
	}

	void* WindowsWindow::GetNativeWindow() const {
		return window_handle_;
	}

	void WindowsWindow::Init(const WindowProperties& props) {
		properties_.title = props.title;
		properties_.width = props.width;
		properties_.height = props.height;

		LF_CORE_INFO("Creating window {0} ({1}, {2})", props.title, props.width, props.height);

		if (glfw_initialized) {
			glfwTerminate();
		}

		const int success = glfwInit();
		LF_CORE_ASSERT(success, "Could not intialize GLFW!");
		glfwSetErrorCallback(GLFWErrorCallback);
		glfw_initialized = true;

		window_handle_ = glfwCreateWindow(static_cast<int>(props.width), static_cast<int>(props.height),
		                                  properties_.title.c_str(), nullptr, nullptr);


		glfwSetWindowUserPointer(window_handle_, &properties_);

		// Set GLFW callbacks
		glfwSetWindowSizeCallback(window_handle_, [](GLFWwindow* window, int width, int height) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
			data.width = width;
			data.height = height;

			WindowResizeEvent event(width, height);
			data.event_callback(event);
		});

		glfwSetWindowCloseCallback(window_handle_, [](GLFWwindow* window) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));
			WindowCloseEvent event;
			data.event_callback(event);
		});

		glfwSetKeyCallback(window_handle_, [](GLFWwindow* window,int key, int scan_code, const int action, int mods) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

			switch (action) {
				case GLFW_PRESS: {
					KeyPressedEvent event(key, 0);
					data.event_callback(event);
					break;
				}
				case GLFW_RELEASE: {
					KeyReleasedEvent event(key);
					data.event_callback(event);
					break;
				}
				case GLFW_REPEAT: {
					KeyPressedEvent event(key, 1);
					data.event_callback(event);
					break;
				}
				default: ;
			}
		});

		glfwSetCharCallback(window_handle_, [](GLFWwindow* window, const unsigned int keycode) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

			KeyTypedEvent event(keycode);
			data.event_callback(event);
		});

		glfwSetMouseButtonCallback(window_handle_, [](GLFWwindow* window, int button, int action, int mods) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

			switch (action) {
				case GLFW_PRESS: {
					MouseButtonPressedEvent event(button);
					data.event_callback(event);
					break;
				}
				case GLFW_RELEASE: {
					MouseButtonReleasedEvent event(button);
					data.event_callback(event);
					break;
				}
			}
		});

		glfwSetScrollCallback(window_handle_, [](GLFWwindow* window, const double x_offset, const double y_offset) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

			MouseScrolledEvent event(static_cast<float>(x_offset), static_cast<float>(y_offset));
			data.event_callback(event);
		});

		glfwSetCursorPosCallback(window_handle_, [](GLFWwindow* window, const double xPos, const double y_pos) {
			WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

			MouseMovedEvent event(static_cast<float>(xPos), static_cast<float>(y_pos));
			data.event_callback(event);
		});
	}

	void WindowsWindow::Shutdown() {
		glfwDestroyWindow(window_handle_);
		glfwTerminate();
	}

}
