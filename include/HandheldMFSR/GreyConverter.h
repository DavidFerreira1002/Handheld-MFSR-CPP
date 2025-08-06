#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <string>

namespace HandheldMFSR {

/// Grey‐conversion methods
enum class GreyMethod { Decimate, FFT };

/// Convert a full‐resolution raw image on the GPU (H×W floats)
/// into a grey image by one of two methods:
///  - Decimate: average each non‐overlapping 2×2 block → (H/2)×(W/2)
///  - FFT: perform 2D FFT, zero out the four 1/4‐corner bands in frequency,
///         inverse FFT, take real part → H×W
///
/// d_in: pointer to H*W float32 array on device
/// H, W: image dimensions (for Decimate, both must be even)
/// method: either GreyMethod::Decimate or GreyMethod::FFT
///
/// Returns: a thrust::device_vector<float> of the output grey image.
thrust::device_vector<float> computeGrey(
    const float* d_in, int H, int W, GreyMethod method);

} // namespace HandheldMFSR
