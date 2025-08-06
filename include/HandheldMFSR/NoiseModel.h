#pragma once
#include <vector>

namespace HandheldMFSR {

/// Holds noise‐curves for a given (alpha, beta) noise model.
struct NoiseModel {
    float alpha = 1.0f;           ///< shot‐noise coefficient
    float beta  = 0.0f;           ///< read‐noise variance
    std::vector<float> stdCurve;  ///< indexed by id_noise=0…maxIndex
    std::vector<float> diffCurve; ///< same length as stdCurve
};

/// Monte Carlo generator for noise curves (port of run_fast_MC).
/// You can either compute on‐the‐fly or load precomputed tables.
class MonteCarlo {
public:
    /// Runs the “fast” Monte Carlo for (alpha,beta) and returns populated curves.
    /// maxIndex controls how many brightness levels (id_noise) to cover.
    static NoiseModel compute(float alpha, float beta, int maxIndex = 5000);
};

} // namespace HandheldMFSR
