#include "pch.h"
#include "Application.h"

#include "Lift/Events/ApplicationEvent.h"
#include "Lift/Log.h"

namespace Lift {

	Application::Application()	{
	}


	Application::~Application() {
	}

	void Application::Run() {
		WindowResizeEvent e(1280, 720);
		if(e.IsInCategory(EventCategoryApplication)) {
			LF_CORE_TRACE(e);
		}
		if(e.IsInCategory(EventCategoryInput)) {
			LF_CORE_TRACE(e);
		}
		while(true);
	}
}
