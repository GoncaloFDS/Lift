#pragma once

#include <string>

struct ShaderProgramSource {
    std::string vertex_source;
    std::string fragment_source;
};

namespace lift {
class Shader {
public:
    Shader(const std::string& file_path);
    ~Shader();

    void bind() const;
    static void unbind() ;

    void setUniform1I(const std::string& name, int value);
    void SetUniform1f(const std::string& name, float value);

    static void setTexImage2D(const uint32_t width, const uint32_t height);

private:
    [[nodiscard]] static auto parseShader(const std::string& file_path) -> ShaderProgramSource ;
    static auto createShader(const std::string& vertex_source, const std::string& fragment_source) -> unsigned int;
    static auto compileShader(const unsigned int type, const std::string& source) -> unsigned;
    auto getUniformLocation(const std::string& name) -> int;

    uint32_t renderer_id_;
    std::string file_path_;
    std::unordered_map<std::string, int> uniform_location_cache_;
};
}
