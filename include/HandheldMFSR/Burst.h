#pragma once

#include <string>
#include <array>
#include <cuda_runtime.h>

namespace HandheldMFSR {

/// Holds all relevant tags extracted from the reference DNG:
///   • ISO between [100,3200]
///   • white/black levels per channel
///   • 2×2 Bayer CFA pattern
///   • per-channel white-balance multipliers
///   • 3×3 camera color matrix (XYZ→camera)
struct BurstTags {
    int ISO = 100;                                      // clipped to [100,3200]
    int white_level = 1;                                // max code value
    std::array<int,4> black_levels = {0,0,0,0};         // order: R, G1, B, G2
    std::array<float,4> white_balance = {1,1,1,1};      // same order
    // Default to RGGB (R=0, G=1, B=2, G=1)
    std::array<std::array<int,2>,2> CFA = {{{0,1},{1,2}}}; 
                                                        // 0=R,1=G,2=B
    std::array<std::array<float,3>,3> xyz2cam = {};     // row-major 3×3
};

/// Burst holds one reference frame (H×W) plus N−1 comparison frames (N×H×W)
/// on the GPU, together with all DNG metadata needed for normalization.
class Burst {
public:
    Burst();
    ~Burst();

    /// Load & decode all .dng/.arw/.cr2/.nef in `folderPath`.
    /// After this call:
    ///   • getFrameCount() == N  (including ref frame)
    ///   • getHeight(), getWidth() set
    ///   • getTags() populated
    ///   • refData(): device pointer to H*W floats (normalized [0,1])
    ///   • compData(): device pointer to N*H*W floats
    ///
    /// Throws std::runtime_error on I/O or CUDA errors.
    void loadFromDisk(const std::string& folderPath);

    // Dimensions
    int getFrameCount() const { return N_; }
    int getHeight()     const { return H_; }
    int getWidth()      const { return W_; }

    // Metadata & buffers
    const BurstTags& getTags()  const { return tags_; }
    const float*     refData()  const { return d_refData_; }
    const float*     compData() const { return d_compData_; }

private:
    int N_ = 0, H_ = 0, W_ = 0;
    BurstTags tags_;

    float* d_refData_  = nullptr;  // cudaMalloc’d H_*W_
    float* d_compData_ = nullptr;  // cudaMalloc’d N_*H_*W_
};

} // namespace HandheldMFSR
