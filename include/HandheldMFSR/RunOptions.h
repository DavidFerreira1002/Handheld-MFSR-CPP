#pragma once
#include <vector>
#include <string>

// BlockMatchingParams
struct BlockMatchingParams {
    std::vector<int>    factors;
    std::vector<int>    tileSizes;
    std::vector<int>    searchRadia;
    std::vector<std::string> distances;
};

// KanadeParams
struct KanadeTuning {
    int   kanadeIter;
    float sigmaBlur;
};

// RobustnessParams
struct RobustnessTuning {
    float t, s1, s2, Mt;
};
struct RobustnessParams {
    bool on;
    RobustnessTuning tuning;
};

// MergingParams
struct MergingTuning {
    float k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink;
};
struct MergingParams {
    std::string kernel;
    MergingTuning tuning;
};

// AccumulatedRobustnessParams
struct MedianParams { bool on; int radiusMax, maxFrameCount; };
struct GaussParams  { bool on; float sigmaMax; int maxFrameCount; };
struct MergeParams  { bool on; int radMax, maxMultiplier, maxFrameCount; };
struct AccumRobustParams {
    MedianParams median;
    GaussParams  gauss;
    MergeParams  merge;
};

// PostProcessingParams
struct PostSharpening { float radius, amount; };
struct PostProcessingParams {
    bool on;
    bool doColorCorr, doToneMapping, doGamma, doSharpening, doDevignette;
    PostSharpening sharpening;
};

// Finally:
struct PipelineParams {
    int scale;
    std::string mode, greyMethod;
    bool debug;
    BlockMatchingParams blockMatching;
    KanadeTuning        kanade;
    RobustnessParams    robustness;
    MergingParams       merging;
    AccumRobustParams   accumRobust;
    PostProcessingParams post;
};
