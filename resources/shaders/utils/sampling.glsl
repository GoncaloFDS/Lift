void sampleLight(inout vec3 ro, inout vec3 ld, inout float light_dist, inout float weight, uint seed) {
    Light light = lights_[0];
    const vec3 light_pos = light.corner.xyz + light.v1.xyz * randomFloat(seed) + light.v2.xyz * randomFloat(seed);
    ld = light_pos - ro;
    light_dist = length(ld);
    ld = normalize(ld);
    const float n_dot_l = dot(ray_.normal, ld);
    const float ln_dot_l = -dot(light.normal.xyz, ld);
    const float A = length(cross(light.v1.xyz, light.v2.xyz));
    weight = n_dot_l * ln_dot_l * A / (c_pi * light_dist * light_dist);
//    weight = clamp(weight, 0.0, 1.0);// remove fireflies
}

void sampleBRDF(inout vec3 ray_dir, inout vec3 ray_pos, inout vec3 throughput, inout bool specular_bounce, inout uint seed) {
    float pdf = 1.0f;
    float do_specular = 0.0f;
    float do_refraction = 0.0f;

    float specular_chance = ray_.mat.specular_chance;
    float refraction_chance = ray_.mat.refraction_chance;

    if (specular_chance > 0.0f) {
        specular_chance = fresnelReflectAmount(
        ray_.from_inside ? ray_.mat.IOR : 1.0,
        !ray_.from_inside ? ray_.mat.IOR : 1.0,
        ray_dir,
        ray_.normal,
        ray_.mat.specular_chance,
        1.0f
        );

        float chance_multiplier = (1.0f - specular_chance) / (1.0f - ray_.mat.specular_chance);
        refraction_chance *= chance_multiplier;
    }

    float ray_select_roll = randomFloat(seed);

    if (specular_chance > 0.0f && ray_select_roll < specular_chance) {
        specular_bounce = true;
        do_specular = 1.0f;
        pdf = specular_chance;
    }
    else if (refraction_chance > 0.0f && ray_select_roll < specular_chance + refraction_chance) {
        specular_bounce = true;
        do_refraction = 1.0f;
        pdf = refraction_chance;
    }
    else {
        specular_bounce = false;
        pdf = 1.0f - (specular_chance + refraction_chance);
    }

    pdf = max(pdf, 0.001f);

    // update the ray position
    if (do_refraction == 1.0f) {
        ray_pos = (ray_pos + ray_dir * ray_.t) - ray_.normal * 0.01f;
    }
    else {
        ray_pos = (ray_pos + ray_dir * ray_.t) + ray_.normal * 0.01f;
    }

    // calculate new ray direction
    vec3 diffuse_ray_dir = normalize(ray_.normal + randomUnitVector(seed));

    vec3 specular_ray_dir = reflect(ray_dir, ray_.normal);
    specular_ray_dir = normalize(mix(specular_ray_dir, diffuse_ray_dir, ray_.mat.specular_roughness * ray_.mat.specular_roughness));

    vec3 refraction_ray_dir = refract(ray_dir, ray_.normal, ray_.from_inside ? ray_.mat.IOR : 1.0f / ray_.mat.IOR);
    refraction_ray_dir = normalize(mix(refraction_ray_dir, normalize(-ray_.normal + randomUnitVector(seed)), ray_.mat.refraction_roughness * ray_.mat.refraction_roughness));

    ray_dir = mix(diffuse_ray_dir, specular_ray_dir, do_specular);
    ray_dir = mix(ray_dir, refraction_ray_dir, do_refraction);

    if (ray_.from_inside) {
        throughput *= exp(-ray_.mat.refraction_color * ray_.t);
    }

    if (do_refraction == 0.0f) {
        throughput *= mix(ray_.mat.albedo, ray_.mat.specular_color, do_specular);
    }
    throughput /= pdf;
}

bool isOccluded(in vec3 ro, in vec3 rd, in float dist) {
    shadow_ray_ = true;
    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    traceRayEXT(scene_, flags, 0xFF, 0, 0, 1, ro, 0.005, rd, dist, 2);
    return shadow_ray_;
}
