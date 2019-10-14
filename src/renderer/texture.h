#pragma once

#include <vector_types.h>
namespace lift {

enum class TextureType {
    NONE = 0,
    DIFFUSE
};

struct Texture {
    unsigned int id{};
    ivec2 resolution{};

    Texture(const std::string& path);
    Texture();
    void bind(unsigned int slot = 0) const;
    void unbind();
    void setData();
    auto data() { return data_.data(); }

    void resize(const ivec2& size) {
        resolution = size;
        data_.resize(size.x * size.y);
    }

 private:
    std::vector<uchar4> data_;
};
}
