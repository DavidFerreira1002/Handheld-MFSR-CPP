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

/// Build an L-level pure-decimation pyramid.
///  - d_in: pointer to the full-res H0×W0 float image on device.
///  - factors: e.g. {1,2,4,…}.  Each level ℓ is (H0/factors[ℓ])×(W0/factors[ℓ]).
/// Returns a vector of length factors.size(), each a device_vector<float> of size h×w.
std::vector<thrust::device_vector<float>>
buildDecimationPyramid(const float* d_in,
                       int H0, int W0,
                       const std::vector<int>& factors);

} // namespace HandheldMFSR
