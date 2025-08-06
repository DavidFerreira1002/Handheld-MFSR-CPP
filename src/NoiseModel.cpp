// src/NoiseModel.cpp
#include "HandheldMFSR/NoiseModel.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace HandheldMFSR {

// Constants matching the Python script
static constexpr int N_PATCHES            = 100000;
static constexpr int N_BRIGHTNESS_LEVELS = 1000;
static constexpr int MAX_INDEX           = N_BRIGHTNESS_LEVELS; // 0…1000
static constexpr int TOL                 = 3;

/// Compute xmin, xmax per get_non_linearity_bound()
static void getNonLinearityBound(float alpha, float beta,
                                 float& xmin, float& xmax) {
    float tol_sq = float(TOL) * TOL;
    xmin = 0.5f * tol_sq * (alpha + std::sqrt(tol_sq*alpha*alpha + 4.f*beta));
    float a = 2.f + tol_sq*alpha;
    float b = 1.f + tol_sq*beta;
    xmax = (a - std::sqrt(a*a - 4.f*b)) * 0.5f;
}

/// Runs the equivalent of unitary_MC in Python for one brightness level b
/// Returns: pair(diff_mean, std_mean)
static std::pair<float,float> unitaryMC(float alpha, float beta, float b) {
    thread_local static std::mt19937_64 gen{std::random_device{}()};
    const float sigma_base = std::sqrt(alpha*b + beta);
    std::normal_distribution<float> dist(0.f, sigma_base);

    double sum_std  = 0.0;
    double sum_diff = 0.0;

    // Temporary arrays
    float patch1[9], patch2[9];

    for (int p = 0; p < N_PATCHES; ++p) {
        // Re-initialize each patch to constant brightness
        for (int i = 0; i < 9; ++i) {
            patch1[i] = b;
            patch2[i] = b;
        }

        // Add noise + clip
        for (int i = 0; i < 9; ++i) {
            patch1[i] = std::clamp(patch1[i] + dist(gen), 0.0f, 1.0f);
            patch2[i] = std::clamp(patch2[i] + dist(gen), 0.0f, 1.0f);
        }

        // Compute means
        float mean1 = 0.f, mean2 = 0.f;
        for (int i = 0; i < 9; ++i) {
            mean1 += patch1[i];
            mean2 += patch2[i];
        }
        mean1 /= 9.f;
        mean2 /= 9.f;

        // Compute std-dev
        float var1 = 0.f, var2 = 0.f;
        for (int i = 0; i < 9; ++i) {
            float d1 = patch1[i] - mean1;
            float d2 = patch2[i] - mean2;
            var1 += d1*d1;
            var2 += d2*d2;
        }
        float std1 = std::sqrt(var1 / 9.f);
        float std2 = std::sqrt(var2 / 9.f);
        float std_mean = 0.5f * (std1 + std2);

        // Mean difference
        float diff_mean = std::fabs(mean1 - mean2);

        sum_std  += std_mean;
        sum_diff += diff_mean;
    }

    return { float(sum_diff / N_PATCHES), float(sum_std / N_PATCHES) };
}

/// Runs MC on an array of brightness levels
static void regularMC(const std::vector<float>& b_array,
                      float alpha, float beta,
                      std::vector<float>& outDiffs,
                      std::vector<float>& outSigmas) {
    int M = int(b_array.size());
    outDiffs.resize(M);
    outSigmas.resize(M);
    for (int i = 0; i < M; ++i) {
        auto [d, s] = unitaryMC(alpha, beta, b_array[i]);
        outDiffs[i]  = d;
        outSigmas[i] = s;
    }
}

/// Linear interpolation of interior points per interp_MC()
static void interpMC(const std::vector<float>& b_array,
                     float sigma_min, float sigma_max,
                     float diff_min,  float diff_max,
                     std::vector<float>& outSigmas,
                     std::vector<float>& outDiffs) {
    int M = int(b_array.size());
    outSigmas.resize(M - 2);
    outDiffs .resize(M - 2);

    float b0 = b_array.front();
    float bN = b_array.back();
    for (int i = 1; i < M-1; ++i) {
        float norm_b = (b_array[i] - b0) / (bN - b0);
        float sq_sig = norm_b*(sigma_max*sigma_max - sigma_min*sigma_min) + sigma_min*sigma_min;
        float sq_dif = norm_b*(diff_max* diff_max  - diff_min* diff_min ) + diff_min* diff_min;
        outSigmas[i-1] = std::sqrt(sq_sig);
        outDiffs [i-1] = std::sqrt(sq_dif);
    }
}

NoiseModel MonteCarlo::compute(float alpha, float beta, int /*maxIndex*/) {
    NoiseModel nm;
    nm.alpha = alpha;
    nm.beta  = beta;
    nm.stdCurve .assign(MAX_INDEX+1, 0.f);
    nm.diffCurve.assign(MAX_INDEX+1, 0.f);

    // Compute non-linearity bounds
    float xmin, xmax;
    getNonLinearityBound(alpha, beta, xmin, xmax);

    // Map to index space
    int imin = std::min(MAX_INDEX,
                int(std::ceil(xmin * N_BRIGHTNESS_LEVELS)) + 1);
    int imax = std::max(0,
                int(std::floor(xmax * N_BRIGHTNESS_LEVELS)) - 1);

    // Brightness vector [0…1] in steps of 1/1000
    std::vector<float> brightness(MAX_INDEX+1);
    for (int i = 0; i <= MAX_INDEX; ++i)
        brightness[i] = float(i) / N_BRIGHTNESS_LEVELS;

    if (imin > MAX_INDEX) {
        // Fallback to regular MC over all levels
        regularMC(brightness, alpha, beta, nm.diffCurve, nm.stdCurve);
        return nm;
    }

    // Prepare non-linear brightness set
    std::vector<float> nl_b;
    nl_b.reserve((imin+1) + (MAX_INDEX-imax+1));
    // [0…imin]
    for (int i = 0; i <= imin; ++i) nl_b.push_back(brightness[i]);
    // [imax…MAX_INDEX]
    for (int i = imax; i <= MAX_INDEX; ++i) nl_b.push_back(brightness[i]);

    // Run MC on non-linear regions
    std::vector<float> sig_nl, dif_nl;
    regularMC(nl_b, alpha, beta, dif_nl, sig_nl);

    // Scatter results back
    // first imin+1 values
    for (int i = 0; i <= imin; ++i) {
        nm.stdCurve[i]  = sig_nl[i];
        nm.diffCurve[i] = dif_nl[i];
    }
    // last (MAX_INDEX-imax+1) values
    int tailCount = MAX_INDEX - imax + 1;
    for (int t = 0; t < tailCount; ++t) {
        nm.stdCurve[imax + t]  = sig_nl[(imin+1) + t];
        nm.diffCurve[imax + t] = dif_nl[(imin+1) + t];
    }

    // Interpolate the middle [imin…imax]
    // brightness_l = brightness[imin-1 … imax+1]
    std::vector<float> b_l;
    for (int i = imin-1; i <= imax+1; ++i)
        b_l.push_back(brightness[i]);

    std::vector<float> sig_l, dif_l;
    interpMC(b_l,
             nm.stdCurve[imin],
             nm.stdCurve[imax],
             nm.diffCurve[imin],
             nm.diffCurve[imax],
             sig_l, dif_l);

    // Copy interpolated interior back: indices imin…imax inclusive
    for (int i = imin; i <= imax; ++i) {
        nm.stdCurve[i]  = sig_l[i - imin];
        nm.diffCurve[i] = dif_l[i - imin];
    }

    return nm;
}

} // namespace HandheldMFSR
