#include "HandheldMFSR/GreyConverter.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>

static bool approx(float a, float b, float eps=1e-5f) {
    return fabs(a - b) < eps;
}

int main() {
    using namespace HandheldMFSR;

    // 1) Decimate path test (4×4 → 2×2)
    {
        int H = 4, W = 4;
        std::vector<float> in = {
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,
            13, 14, 15, 16
        };
        // Upload to device
        float* d_in = nullptr;
        cudaMalloc(&d_in, H*W*sizeof(float));
        cudaMemcpy(d_in, in.data(), H*W*sizeof(float), cudaMemcpyHostToDevice);

        // Compute
        auto d_out = computeGrey(d_in, H, W, GreyMethod::Decimate);
        cudaFree(d_in);

        // Copy back to host
        thrust::host_vector<float> h_out = d_out;

        // Expected
        std::vector<float> expected = {3.5f, 5.5f, 11.5f, 13.5f};
        for (int i = 0; i < 4; ++i) {
            assert(approx(h_out[i], expected[i]));
        }
        std::cout << "Decimate test passed.\n";
    }

    // 2) FFT path test (constant image remains constant)
    {
        int H = 8, W = 8;
        size_t N = H*W;
        float v = 2.75f;
        // Prepare host
        std::vector<float> in(N, v);

        // Upload
        float* d_in = nullptr;
        cudaMalloc(&d_in, N*sizeof(float));
        cudaMemcpy(d_in, in.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        // Compute
        auto d_out = computeGrey(d_in, H, W, GreyMethod::FFT);
        cudaFree(d_in);

        // Copy back
        thrust::host_vector<float> h_out = d_out;

        // Verify
        for (size_t i = 0; i < N; ++i) {
            assert(approx(h_out[i], v, 1e-3f));
        }
        std::cout << "FFT test passed.\n";
    }

    std::cout << "TestGreyConverter all tests passed.\n";
    return 0;
}
