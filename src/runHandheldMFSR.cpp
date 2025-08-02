// HandheldMFSR.h

#pragma once
#include <string>
#include <optional>

struct RunOptions {
    int verbose = 1;
};

struct RobustnessParams {
    bool on               = true;
    bool denoising_on     = true;
    float t               = 0.12f;
    float s1              = 2.0f;
    float s2              = 12.0f;
    float Mt              = 0.8f;
};

struct PostProcessingParams {
    bool on                   = true;
    bool do_sharpening        = true;
    bool do_tonemapping       = true;
    bool do_gamma             = true;
    bool do_color_correction  = true;
    bool do_devignette        = false;
    float radius              = 3.0f;
    float amount              = 1.5f;
};

struct MergingParams {
    std::string kernel        = "handheld";
    float k_stretch           = 4.0f;
    float k_shrink            = 2.0f;
    std::optional<float> k_detail;
    std::optional<float> k_denoise;
};

struct KanadeParams {
    int ICA_iter              = 3;
};

struct PipelineParams {
    int scale                              = 2;
    MergingParams merging;
    RobustnessParams robustness;
    KanadeParams kanade;
    PostProcessingParams post_processing;
};

/// Load a burst from `inputPath`, run the SR pipeline on GPU, and return
/// an H×W×3 float32 image in [0,1]. 
///
/// - inputPath: folder containing RAW frames (e.g. DNG)
/// - opts:     verbosity, etc.
/// - params:   full pipeline configuration
///
/// Throws on I/O or CUDA errors.
std::vector<float> runHandheldMFSR(
    const std::string& inputPath,
    const RunOptions& opts,
    const PipelineParams& params);