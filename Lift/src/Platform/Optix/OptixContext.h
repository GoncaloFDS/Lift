#pragma once

namespace lift {
	class OptixContext {
	public:
		OptixContext() = default;

		static void PrintInfo();
		static optix::Context& Get() { return context_; }
		static optix::Context& Create();

	private:
		static optix::Context context_;
	};
}
