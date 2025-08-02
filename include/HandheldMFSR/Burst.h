#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace HandheldMFSR {

/// Burst holds N Bayer-pattern frames on the GPU, plus the noise-profile α,β.
class Burst {
public:
    Burst();
    ~Burst();

    /// Scan folderPath for RAW/DNG files, decode them via LibRaw,
    /// and upload a contiguous CFA array (size N×H×W) to the GPU.
    void loadFromDisk(const std::string& folderPath);

    /// Number of frames in the burst
    int getFrameCount() const { return N_; }
    /// Height of each frame
    int getHeight()     const { return H_; }
    /// Width of each frame
    int getWidth()      const { return W_; }

    /// Noise-profile parameters (read from EXIF or defaults)
    float getAlpha() const { return alpha_; }
    float getBeta()  const { return beta_;  }

    /// Pointer to device memory holding N*H*W float32 CFA samples
    /// Laid out as frame-major: data[f*H*W + y*W + x]
    const float* data() const { return d_data_; }

private:
    int   N_      = 0;
    int   H_      = 0;
    int   W_      = 0;
    float alpha_  = 1.0f;
    float beta_   = 0.0f;

    float* d_data_ = nullptr;  // cudaMalloc’d buffer of size N_*H_*W_
};

} // namespace HandheldMFSR