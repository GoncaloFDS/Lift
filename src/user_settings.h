#pragma once

struct UserSettings final {
    // Application
    bool benchmark{};

    // Benchmark
    bool benchmarkNextScenes{};
    uint32_t benchmarkMaxTime{};

    // Scene
    int sceneIndex{};

    // Renderer
    bool isRayTraced{};
    bool accumulateRays{};
    uint32_t numberOfSamples{};
    uint32_t numberOfBounces{};
    uint32_t maxNumberOfSamples{};

    // Camera
    float fieldOfView{};
    float aperture{};
    float focusDistance{};
    bool gammaCorrection{};

    // UI
    bool showSettings{};
    bool showOverlay{};

    [[nodiscard]] bool requiresAccumulationReset(const UserSettings& prev) const {
        return isRayTraced != prev.isRayTraced ||
            accumulateRays != prev.accumulateRays ||
            numberOfBounces != prev.numberOfBounces ||
            fieldOfView != prev.fieldOfView ||
            aperture != prev.aperture ||
            focusDistance != prev.focusDistance;
    }
};
