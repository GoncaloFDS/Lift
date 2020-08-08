struct RayPayload {
    float t;
    vec3 normal;
    Material mat;
    bool missed;
    bool from_inside;
};
