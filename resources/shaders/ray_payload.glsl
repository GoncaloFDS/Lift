struct RayPayload {
    vec4 ColorAndDistance;// rgb + t
    vec4 ScatterDirection;// xyz + w (is scatter needed)
    uint random_seed;
};

struct PerRayData {
    vec3 result;
    vec3 radiance;
    vec3 attenuation;
    vec3 origin;
    vec3 direction;
    uint seed;
    int depth;
    int done;
};
