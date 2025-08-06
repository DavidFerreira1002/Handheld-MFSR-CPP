#include "HandheldMFSR/Pyramid.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>

static bool approx(float a, float b, float eps=1e-3f) {
    return fabs(a - b) < eps;
}

int main() {
    using namespace HandheldMFSR;

    // Test: 8×8 impulse at (0,0) → pyramid factors {1,2,4}
    int H = 8, W = 8;
    std::vector<float> in(H*W, 0.0f);
    in[0] = 1.0f;  // Dirac impulse

    // Upload
    float* d_in = nullptr;
    cudaMalloc(&d_in, H*W*sizeof(float));
    cudaMemcpy(d_in, in.data(), H*W*sizeof(float), cudaMemcpyHostToDevice);

    PyramidParams params;
    params.factors = {1,2,4};

    auto levels = buildGaussianPyramid(d_in, H, W, params);
    cudaFree(d_in);

    // Expect 3 levels: sizes (2,2), (4,4), (8,8) reversed → [ coarse(4x4), mid(2x2), fine(8x8) ]
    assert(levels.size() == 3);
    std::vector<std::pair<int,int>> shapes = {{2,2}, {4,4}, {8,8}}; // reversed view
    for (int i = 0; i < 3; ++i) {
        int h = shapes[i].first, w = shapes[i].second;
        thrust::host_vector<float> h_lvl = levels[i];
        // Check the impulse survives at [0]
        assert(h_lvl[0] > 0.0f);

        // Sum check:
        float sum = 0.f;
        for (int j = 0; j < h*w; ++j) sum += h_lvl[j];

        if (h == H && w == W) {
            // finest level: should preserve sum ≈1
            assert(approx(sum, 1.0f));
        } else {
            // coarser levels: sum should be positive but < 1
            assert(sum > 0.0f && sum < 1.0f);
        }
    }

    std::cout << "TestPyramid passed!\n";
    return 0;
}
