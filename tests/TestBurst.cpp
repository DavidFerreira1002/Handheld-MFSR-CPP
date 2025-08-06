#include "HandheldMFSR/Burst.h"
#include <cnpy.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

int main() {
    using namespace HandheldMFSR;

    // 1) Load raw test burst (N × H × W) from .npy
    cnpy::NpyArray arr = cnpy::npy_load("tests/data/test_burst.npy");
    assert(arr.word_size == sizeof(float));
    assert(arr.shape.size() == 3);
    int N = static_cast<int>(arr.shape[0]);
    int H = static_cast<int>(arr.shape[1]);
    int W = static_cast<int>(arr.shape[2]);
    const float* rawData = arr.data<float>();

    // 2) Load with our Burst loader (normalizes into [0,1])
    Burst burst;
    burst.loadFromDisk("tests/data/test_burst_folder");
    assert(burst.getFrameCount() == N);
    assert(burst.getHeight()     == H);
    assert(burst.getWidth()      == W);

    const auto& T = burst.getTags();

    // 3) Prepare CPU‐normalized data for frame 0 and frame 1
    auto normalize_pixel = [&](float v, int y, int x) {
        int c = T.CFA[y & 1][x & 1]; 
        float bl = static_cast<float>(T.black_levels[c]);
        float wl = static_cast<float>(T.white_level - T.black_levels[c]);
        // scale by (white_balance[c]/white_balance[1])
        float wb = T.white_balance[c] / T.white_balance[1];
        float nv = (v - bl) / wl;
        nv = nv * wb;
        return std::min(std::max(nv, 0.0f), 1.0f);
    };

    size_t frameSize = size_t(H) * W;
    std::vector<float> cpuRef(frameSize), cpuComp(frameSize);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = size_t(y) * W + x;
            cpuRef[idx]  = normalize_pixel(rawData[0 * frameSize + idx], y, x);
            cpuComp[idx] = normalize_pixel(rawData[1 * frameSize + idx], y, x);
        }
    }

    // 4) Copy GPU data back to host
    std::vector<float> gpuRef(frameSize), gpuComp(frameSize);
    cudaMemcpy(gpuRef.data(),  burst.refData(),  frameSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuComp.data(), burst.compData() + frameSize, frameSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 5) Compare with a small tolerance
    const float eps = 1e-4f;
    float maxDiffRef = 0.0f, maxDiffComp = 0.0f;
    for (size_t i = 0; i < frameSize; ++i) {
        float d0 = std::fabs(gpuRef[i]  - cpuRef[i]);
        float d1 = std::fabs(gpuComp[i] - cpuComp[i]);
        maxDiffRef  = std::max(maxDiffRef, d0);
        maxDiffComp = std::max(maxDiffComp, d1);
        if (d0 > eps) {
            std::cerr << "Reference mismatch at [" << (i/W) << "," << (i%W) 
                      << "]: gpu=" << gpuRef[i] << " cpu=" << cpuRef[i] << "\n";
            return 1;
        }
        if (d1 > eps) {
            std::cerr << "Comp frame mismatch at [" << (i/W) << "," << (i%W) 
                      << "]: gpu=" << gpuComp[i] << " cpu=" << cpuComp[i] << "\n";
            return 2;
        }
    }

    std::cout << "TestBurst passed. maxDiffRef=" << maxDiffRef 
              << ", maxDiffComp=" << maxDiffComp << "\n";
    return 0;
}
