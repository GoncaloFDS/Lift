#pragma once

struct UserSettings final {
    // Benchmark
    bool benchmark {};
    bool benchmark_next_scenes {};
    uint32_t benchmark_max_time {};

    // Scene
    int scene_index {};
    int algorithm_index {};

    bool is_denoised {};
    bool accumulate_rays {};
    uint32_t number_of_samples {};
    uint32_t number_of_bounces {};
    uint32_t max_number_of_samples {};

    // debug
    bool debug_normals {};

    // Camera
    float fov {};
    float aperture {};
    float focus_distance {};
    float camera_move_speed {};
    float camera_mouse_speed {};

    // Post processing
    bool gamma_correction {};
    bool tone_map {};
    float exposure {};

    // UI
    bool show_settings {};
    bool show_overlay {};

    [[nodiscard]] bool requiresAccumulationReset(const UserSettings& prev) const {
        return accumulate_rays != prev.accumulate_rays ||
               number_of_samples != prev.number_of_samples ||
               number_of_bounces != prev.number_of_bounces ||
               fov != prev.fov ||
               aperture != prev.aperture ||
               focus_distance != prev.focus_distance ||
               debug_normals != prev.debug_normals;
    }

};
