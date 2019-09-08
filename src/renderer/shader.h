#pragma once

#include <string>

struct ShaderProgramSource {
    std::string vertex_source;
    std::string fragment_source;
};

namespace lift {
class Shader {
public:
    Shader(const std::string &file_path);
    ~Shader();

    void bind() const;
    void unbind() const;

    void setUniform1I(const std::string &name, int value);
    void SetUniform1f(const std::string &name, float value);

    static void setTexImage2D(const uint32_t width, const uint32_t height);

private:
    ShaderProgramSource parseShader(const std::string &file_path) const;
    static unsigned int createShader(const std::string &vertex_source, const std::string &fragment_source);
    static unsigned compileShader(const unsigned int type, const std::string &source);
    int getUniformLocation(const std::string &name);

    uint32_t renderer_id_;
    std::string file_path_;
    std::unordered_map<std::string, int> uniform_location_cache_;
};
}
