#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include "GaussianKernel1D.h"

namespace HandheldMFSR {

/// Coarse-to-fine pyramid parameters.
struct PyramidParams {
    std::vector<int> factors;   // e.g. {1,2,4,4}
    float            sigmaScale = 0.5f; // kernel σ = factor * sigmaScale
};

/// Builds a Gaussian pyramid of d_in (H×W) per params.
/// Returns device vectors [coarse ... fine], each flattened row-major.
std::vector<thrust::device_vector<float>>
buildGaussianPyramid(const float* d_in, int H, int W,
                     const PyramidParams& params);

} // namespace HandheldMFSR
