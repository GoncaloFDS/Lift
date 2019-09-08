#include "pch.h"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "renderer/BufferView.h"
#include "scene/Scene.h"

template<typename T>
lift::BufferView<T> bufferViewFromGltf(const tinygltf::Model &model, lift::Scene *scene, const int32_t accessor_idx) {
    if (accessor_idx==-1)
        return lift::BufferView<T>();

    const auto &gltf_accessor = model.accessors[accessor_idx];
    const auto &gltf_buffer_view = model.bufferViews[gltf_accessor.bufferView];

    const int32_t elmt_byte_size =
        gltf_accessor.componentType==TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
        ? 2
        : gltf_accessor.componentType==TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
          ? 4
          : gltf_accessor.componentType==TINYGLTF_COMPONENT_TYPE_FLOAT
            ? 4
            : 0;
    LF_ASSERT(elmt_byte_size, "gltf accessor component type not supported");

    const CUdeviceptr buffer_base = scene->getBuffer(gltf_buffer_view.buffer);
    lift::BufferView<T> buffer_view;
    buffer_view.data = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride = static_cast<uint16_t>(gltf_buffer_view.byteStride);
    buffer_view.count = static_cast<uint32_t>(gltf_accessor.count);
    buffer_view.elmt_byte_size = static_cast<uint16_t>(elmt_byte_size);

    return buffer_view;
}
