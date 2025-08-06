#include "HandheldMFSR/SNR.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace HandheldMFSR {

PipelineParams getParamsForSNR(float SNR_in) {
    // 1) Clip SNR
    float SNR = std::clamp(SNR_in, 6.0f, 30.0f);

    // 2) Compute tile size Ts
    int Ts;
    if (SNR <= 14.0f) {
        Ts = 64;
    } else if (SNR <= 22.0f) {
        Ts = 32;
    } else {
        Ts = 16;
    }
    // Python warns and clamps Ts>32 → 32
    if (Ts > 32) {
        Ts = 32;
        // (We could log a warning here)
    }

    // 3) Merging tuning parameters (linear interpolation)
    // k_detail ∈ [0.25, 0.33], k_denoise ∈ [3,5], D_th ∈ [0.71,0.81], D_tr ∈ [1,1.24]
    float range = 30.0f - 6.0f;
    float factor = (30.0f - SNR) / range;
    float k_detail  = 0.25f + (0.33f - 0.25f) * factor;
    float k_denoise = 3.0f  + ( 5.0f - 3.0f)  * factor;
    float D_th       = 0.71f + (0.81f - 0.71f) * factor;
    float D_tr       = 1.0f  + (1.24f - 1.0f)  * factor;

    // 4) Fill PipelineParams
    PipelineParams p;

    // Top-level
    p.scale      = 1;
    p.mode       = "bayer";    // or enum if you prefer
    p.greyMethod = "FFT";      // not used downstream except for checks
    p.debug      = false;

    // Block matching
    p.blockMatching.factors      = {1,2,4,4};
    p.blockMatching.tileSizes    = {Ts, Ts, Ts, Ts/2};
    p.blockMatching.searchRadia  = {1, 4, 4, 4};
    p.blockMatching.distances    = {"L1","L2","L2","L2"};

    // ICA (Kanade)
    p.kanade.kanadeIter = 3;
    p.kanade.sigmaBlur  = 0.0f;

    // Robustness
    p.robustness.on            = true;
    p.robustness.tuning.t     = 0.12f;
    p.robustness.tuning.s1    = 2.0f;
    p.robustness.tuning.s2    = 12.0f;
    p.robustness.tuning.Mt    = 0.8f;

    // Merging
    p.merging.kernel            = "handheld";
    p.merging.tuning.k_detail   = k_detail;
    p.merging.tuning.k_denoise  = k_denoise;
    p.merging.tuning.D_th       = D_th;
    p.merging.tuning.D_tr       = D_tr;
    p.merging.tuning.k_stretch  = 4.0f;
    p.merging.tuning.k_shrink   = 2.0f;

    // Accumulated-robustness denoiser
    p.accumRobust.median.on      = false;
    p.accumRobust.median.radiusMax = 3;
    p.accumRobust.median.maxFrameCount = 8;

    p.accumRobust.gauss.on        = false;
    p.accumRobust.gauss.sigmaMax  = 1.5f;
    p.accumRobust.gauss.maxFrameCount = 8;

    p.accumRobust.merge.on        = true;
    p.accumRobust.merge.radMax    = 2;
    p.accumRobust.merge.maxMultiplier = 8;
    p.accumRobust.merge.maxFrameCount = 8;

    // Post-processing
    p.post.on             = true;
    p.post.doColorCorr    = true;
    p.post.doToneMapping  = true;
    p.post.doGamma        = true;
    p.post.doSharpening   = true;
    p.post.doDevignette   = false;
    p.post.sharpening.radius = 3.0f;
    p.post.sharpening.amount = 1.5f;

    return p;
}

} // namespace HandheldMFSR