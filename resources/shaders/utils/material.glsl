const uint MaterialLambertian = 0;
const uint MaterialMetallic = 1;
const uint MaterialDielectric = 2;
const uint MaterialIsotropic = 3;
const uint MaterialDiffuseLight = 4;

struct Material {
    vec4 diffuse;
    int diffuse_texture;
    float metallic_factor;
    float refraction_index;
    uint shading_model;
};
