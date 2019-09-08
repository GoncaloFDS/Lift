#pragma once
namespace lift {
struct Texture;

class FrameBuffer {
public:
    FrameBuffer();
    ~FrameBuffer();

    void bind() const;
    void unbind();
    void bindTexture(const Texture texture);

private:
    unsigned renderer_id_;
};
}
