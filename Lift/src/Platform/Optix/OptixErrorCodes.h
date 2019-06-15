#pragma once
#include <optix_host.h>

namespace lift {

	static std::string RTresultToString(const RTresult result) {
		switch (result) {
			case RT_SUCCESS: return "Success";
			case RT_TIMEOUT_CALLBACK: return "Timeout callback";
			case RT_ERROR_INVALID_CONTEXT: return "Invalid context";
			case RT_ERROR_INVALID_VALUE: return "Invalid value";
			case RT_ERROR_MEMORY_ALLOCATION_FAILED: return "Memory allocation failed";
			case RT_ERROR_TYPE_MISMATCH: return "Type mismatch";
			case RT_ERROR_VARIABLE_NOT_FOUND: return "Variable not found";
			case RT_ERROR_VARIABLE_REDECLARED: return "Variable redeclared";
			case RT_ERROR_ILLEGAL_SYMBOL: return "Illegal Symbol";
			case RT_ERROR_INVALID_SOURCE: return "Invalid Source";
			case RT_ERROR_VERSION_MISMATCH: return "Version mismatch";
			case RT_ERROR_OBJECT_CREATION_FAILED: return "Object Creation Failed";
			case RT_ERROR_NO_DEVICE: return "No device";
			case RT_ERROR_INVALID_DEVICE: return "Invalid device";
			case RT_ERROR_INVALID_IMAGE: return "Invalid image";
			case RT_ERROR_FILE_NOT_FOUND: return "File not found";
			case RT_ERROR_ALREADY_MAPPED: return "Already mapped";
			case RT_ERROR_INVALID_DRIVER_VERSION: return "Invalid driver version";
			case RT_ERROR_CONTEXT_CREATION_FAILED: return "Context creation failed";
			case RT_ERROR_RESOURCE_NOT_REGISTERED: return "Resource not registered";
			case RT_ERROR_RESOURCE_ALREADY_REGISTERED: return "Resource already registered";
			case RT_ERROR_OPTIX_NOT_LOADED: return "Optix not loaded";
			case RT_ERROR_DENOISER_NOT_LOADED: return "Denoiser not loaded";
			case RT_ERROR_SSIM_PREDICTOR_NOT_LOADED: return "SSIM predicator not loaded";
			case RT_ERROR_DRIVER_VERSION_FAILED: return "Driver version failed";
			case RT_ERROR_DATABASE_FILE_PERMISSIONS: return "Database file permissions";
			case RT_ERROR_LAUNCH_FAILED: return "Launch failed";
			case RT_ERROR_NOT_SUPPORTED: return "Not supported";
			case RT_ERROR_CONNECTION_FAILED: return "Connection failed";
			case RT_ERROR_AUTHENTICATION_FAILED: return "Authentication failed";
			case RT_ERROR_CONNECTION_ALREADY_EXISTS: return "Connection already exists";
			case RT_ERROR_NETWORK_LOAD_FAILED: return "Network Load failed";
			case RT_ERROR_NETWORK_INIT_FAILED: return "Network Init failed";
			case RT_ERROR_CLUSTER_NOT_RUNNING: return "Cluster not running";
			case RT_ERROR_CLUSTER_ALREADY_RUNNING: return "Cluster already running";
			case RT_ERROR_INSUFFICIENT_FREE_NODES: return "Insufficient free nodes";
			case RT_ERROR_INVALID_GLOBAL_ATTRIBUTE: return "Invalid global attribute";
			case RT_ERROR_UNKNOWN: return "Unknown error";
			default: return "RTresult code Unknown";
		}
	}


}
