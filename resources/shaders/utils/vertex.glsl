struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 tex_coords;
    int material_index;
};

Vertex unpackVertex(uint index) {
    const uint vertexSize = 9;
    const uint offset = index * vertexSize;

    Vertex v;
    v.position = vec3(vertices_[offset + 0], vertices_[offset + 1], vertices_[offset + 2]);
    v.normal = vec3(vertices_[offset + 3], vertices_[offset + 4], vertices_[offset + 5]);
    v.tex_coords = vec2(vertices_[offset + 6], vertices_[offset + 7]);
    v.material_index = floatBitsToInt(vertices_[offset + 8]);

    return v;
}
