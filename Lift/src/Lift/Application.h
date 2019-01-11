#pragma once

#include "Core.h"

namespace Lift {

	class LIFT_API Application	{
	public:
		Application();
		virtual ~Application();

		void Run();
	};

	Application* CreateApplication();
}



