#pragma once
#include "buffer_view.h"

namespace lift {
struct GeometryData {
    enum Type {
        TRIANGLE_MESH = 0,
        SPHERE
    };

    struct TriangleMesh {
        BufferView<uint32_t> indices;
        BufferView<float3> positions;
        BufferView<float3> normals;
        BufferView<float2> tex_coords;
    };

    struct Sphere {
        float3 center;
        float radius;
    };

    Type type;

    union {
        TriangleMesh triangle_mesh;
        Sphere sphere;
    };
};
}
