struct Light {
    vec4 corner;
    vec4 v1;
    vec4 v2;
    vec4 normal;
    vec4 emission;
};

struct UniformBufferObject {
    mat4 model_view;
    mat4 projection;
    mat4 model_view_inverse;
    mat4 projection_inverse;
    float aperture;
    float focus_distance;
    uint total_number_of_samples;
    uint number_of_samples;
    uint number_of_bounces;
    uint seed;
    bool gamma_correction;
    bool tone_map;
    float exposure;
    bool has_sky;
    uint frame;
    bool debug_normals;
};
