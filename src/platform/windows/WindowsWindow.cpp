#include "pch.h"

#include "WindowsWindow.h"

#include "events/ApplicationEvent.h"
#include "events/MouseEvent.h"
#include "events/KeyEvent.h"


static bool glfw_initialized = false;

static void GLFWErrorCallback(const int error, const char* description) {
	LF_CORE_ERROR("GLFW Error ({0}): {1}", error, description);
}

lift::Window* lift::Window::Create(const WindowProperties& props) {
	return new WindowsWindow(props);
}

lift::WindowsWindow::WindowsWindow(const WindowProperties& props) {
	WindowsWindow::Init(props);
}

lift::WindowsWindow::~WindowsWindow() {
	WindowsWindow::Shutdown();
}

void lift::WindowsWindow::OnUpdate() {
	glfwPollEvents();
}

unsigned lift::WindowsWindow::Width() const {
	return properties_.width;
}

unsigned lift::WindowsWindow::Height() const {
	return properties_.height;
}

std::pair<int, int> lift::WindowsWindow::GetPosition() const {
	int x, y;
	glfwGetWindowPos(window_handle_, &x, &y);
	return {x, y};
}

void lift::WindowsWindow::SetEventCallback(const EventCallbackFn& callback) {
	properties_.event_callback = callback;
}

void lift::WindowsWindow::SetVSync(const bool enabled) {
	if (enabled)
		glfwSwapInterval(1);
	else
		glfwSwapInterval(0);
	properties_.v_sync = enabled;
}

bool lift::WindowsWindow::IsVSync() const {
	return properties_.v_sync;
}

void* lift::WindowsWindow::GetNativeWindow() const {
	return window_handle_;
}

void lift::WindowsWindow::Init(const WindowProperties& props) {
	properties_.title = props.title;
	properties_.width = props.width;
	properties_.height = props.height;
	properties_.x = props.x;
	properties_.y = props.y;

	LF_CORE_INFO("Creating window {0} ({1}, {2})", props.title, props.width, props.height);

	if (glfw_initialized) {
		glfwTerminate();
	}

	const int success = glfwInit();
	LF_CORE_ASSERT(success, "Could not intialize GLFW!");
	glfwSetErrorCallback(GLFWErrorCallback);
	glfw_initialized = true;
	//glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

	window_handle_ = glfwCreateWindow(static_cast<int>(props.width), static_cast<int>(props.height),
									  properties_.title.c_str(), nullptr, nullptr);



	glfwSetWindowPos(window_handle_, props.x, props.y);
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

	glfwSetWindowIconifyCallback(window_handle_, [](GLFWwindow* window, int iconified) {
		WindowData& data = *static_cast<WindowData*>(glfwGetWindowUserPointer(window));

		WindowMinimizeEvent event(iconified);
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

void lift::WindowsWindow::Shutdown() {
	glfwDestroyWindow(window_handle_);
	glfwTerminate();
}