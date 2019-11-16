#include "pch.h"
#include "util.h"

auto lift::Util::getPtxString(const char* file_name) -> std::string {
    std::string ptx_source;

    const std::ifstream file(file_name);
    if (file.good()) {
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        return source_buffer.str();
    }
    LF_ERROR("Invalid PTX path: {0}", file_name);
    return "Invalid PTX";
}

