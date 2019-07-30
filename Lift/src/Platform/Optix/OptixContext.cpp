#include "pch.h"
#include "OptixContext.h"

optix::Context lift::OptixContext::context_; 

void lift::OptixContext::PrintInfo() {

	unsigned int optix_version;
	rtGetVersion(&optix_version);

	const auto major = optix_version / 10000;
	const auto minor = (optix_version % 10000) / 100;
	const auto micro = optix_version % 100;
	LF_CORE_INFO("");
	LF_CORE_INFO("Optix Info:");
	LF_CORE_INFO("\tVersion: {0}.{1}.{2}", major, minor, micro);

	const auto number_of_devices = optix::Context::getDeviceCount();
	LF_CORE_INFO("\tNumber of Devices = {0}", number_of_devices);

	for (unsigned i = 0; i < number_of_devices; ++i) {
		char name[256];
		context_->getDeviceAttribute(int(i), RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name);
		LF_CORE_INFO("\tDevice {0}: {1}", i, name);

		int compute_capability[2] = {0, 0};
		context_->getDeviceAttribute(int(i), RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(
											   compute_capability), &compute_capability);
		LF_CORE_INFO("\t\tCompute Support: {0}.{1}", compute_capability[0], compute_capability[1]);
	}
}

optix::Context& lift::OptixContext::Create() {
	context_ = optix::Context::create();
	return context_;
}
