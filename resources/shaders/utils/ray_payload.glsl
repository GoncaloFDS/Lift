struct PerRayData {
    vec3 emitted;
    vec3 radiance;
    vec3 attenuation;
    vec3 origin;
    vec3 direction;
    uint seed;
    int depth;
    bool done;
    bool countEmitted;
};
