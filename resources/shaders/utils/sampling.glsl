const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;

float remap(float s, float a1, float a2, float b1, float b2) {
    return b1 + (s-a1)*(b2-b1)/(a2-a1);
}

vec3 remap(vec3 v, float a1, float a2, float b1, float b2) {
    v.x = remap(v.x, a1, a2, b1, b2);
    v.y = remap(v.y, a1, a2, b1, b2);
    v.z = remap(v.z, a1, a2, b1, b2);
    return v;
}

float fresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident, float f0, float f90) {
    // Schlick aproximation
    float r0 = (n1-n2) / (n1+n2);
    r0 *= r0;
    float cosX = -dot(normal, incident);
    if (n1 > n2)
    {
        float n = n1/n2;
        float sinT2 = n*n*(1.0-cosX*cosX);
        // Total internal reflection
        if (sinT2 > 1.0)
        return f90;
        cosX = sqrt(1.0-sinT2);
    }
    float x = 1.0-cosX;
    float ret = r0+(1.0-r0)*x*x*x*x*x;

    // adjust reflect multiplier for object reflectivity
    return mix(f0, f90, ret);
}

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

uint initRandom(vec2 v, uint frame) {
    return uint(uint(v.x) * uint(1973) + uint(v.y) * uint(9277) + frame * uint(26699)) | uint(1);
}

float randomFloat(inout uint seed) {
    return float(wang_hash(seed)) / 4294967296.0;
}

vec3 randomUnitVector(inout uint seed) {
    float z = randomFloat(seed) * 2.0f - 1.0f;
    float a = randomFloat(seed) * c_twopi;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}