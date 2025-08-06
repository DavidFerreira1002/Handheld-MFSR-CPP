#include "HandheldMFSR/SNR.h"
#include <cassert>
#include <iostream>
#include <cmath>

static bool approxEqual(float a, float b, float eps=1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    using namespace HandheldMFSR;

    // Test SNR = 6 → Ts clamped to 32; k_detail ~= 0.33
    {
        auto p = getParamsForSNR(6.0f);
        assert(p.blockMatching.tileSizes[0] == 32);
        float expected = 0.25f + (0.33f - 0.25f)*(30-6)/24.0f; // = 0.33
        assert(approxEqual(p.merging.tuning.k_detail, expected));
    }

    // Test SNR = 14 → Ts=32; k_denoise = 3 + (5-3)*(30-14)/24 = 3 + 2*(16/24)=4.333…
    {
        auto p = getParamsForSNR(14.0f);
        assert(p.blockMatching.tileSizes[0] == 32);
        float expected = 3.0f + 2.0f*(30-14)/24.0f; // 3 + 2*(16/24)=4.3333
        assert(approxEqual(p.merging.tuning.k_denoise, expected));
    }

    // Test SNR = 22 → Ts=32; D_th = 0.71 + (0.81-0.71)*(30-22)/24 = 0.7433…
    {
        auto p = getParamsForSNR(22.0f);
        assert(p.blockMatching.tileSizes[0] == 32);
        float expected = 0.71f + 0.10f*(30-22)/24.0f; // 0.7433
        assert(approxEqual(p.merging.tuning.D_th, expected));
    }

    // Test SNR = 30 → Ts=16; k_detail = 0.25
    {
        auto p = getParamsForSNR(30.0f);
        assert(p.blockMatching.tileSizes[0] == 16);
        assert(approxEqual(p.merging.tuning.k_detail, 0.25f));
    }

    std::cout << "TestGetParams passed!\n";
    return 0;
}
