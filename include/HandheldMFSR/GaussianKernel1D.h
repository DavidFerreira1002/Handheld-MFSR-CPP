#pragma once
#include <vector>

namespace HandheldMFSR {

/// Computes an order-0 Gaussian kernel:
///   radius = floor(4σ + 0.5)
///   kernel[i] = exp(-0.5 * ((i-radius)/σ)^2)
///   then normalize to sum to 1.
/// Returns a vector of length (2*radius+1).
std::vector<float> gaussianKernel1D(float sigma);

} // namespace HandheldMFSR
