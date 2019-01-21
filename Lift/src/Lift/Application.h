#pragma once

#include "Core.h"
#include "Events/Event.h"

namespace Lift {

	class LIFT_API Application	{
	public:
		Application();
		virtual ~Application();

		void Run();
	};

	Application* CreateApplication();
}



