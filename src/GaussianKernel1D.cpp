#include "HandheldMFSR/GaussianKernel1D.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace HandheldMFSR {

std::vector<float> gaussianKernel1D(float sigma) {
    if (sigma <= 0.0f) throw std::invalid_argument("sigma must be positive");
    int radius = int(std::floor(4.0f * sigma + 0.5f));
    int size   = 2*radius + 1;
    std::vector<float> k(size);
    float sigma2 = sigma * sigma;
    for (int i = 0; i < size; ++i) {
        float x = float(i - radius);
        k[i] = std::exp(-0.5f * x*x / sigma2);
    }
    float sum = std::accumulate(k.begin(), k.end(), 0.0f);
    for (auto& v : k) v /= sum;
    return k;
}

} // namespace HandheldMFSR
