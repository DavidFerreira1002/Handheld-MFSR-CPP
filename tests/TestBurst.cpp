#include "HandheldMFSR/Burst.h"
#include <cnpy.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using namespace HandheldMFSR;

    // 1) Load reference burst from .npy
    cnpy::NpyArray arr = cnpy::npy_load("tests/data/test_burst.npy");
    assert(arr.word_size == sizeof(float));
    assert(arr.shape.size() == 3);
    int N = arr.shape[0];
    int H = arr.shape[1];
    int W = arr.shape[2];
    const float* refData = reinterpret_cast<const float*>(arr.data<float>());

    // 2) Use our Burst loader to upload the same files
    //    (ensure that `tests/data/test_burst_folder` exists
    //     containing the original DNGs)
    Burst burst;
    burst.loadFromDisk("tests/data/test_burst_folder");

    assert(burst.getFrameCount() == N);
    assert(burst.getHeight()     == H);
    assert(burst.getWidth()      == W);

    // 3) Copy device data back to host
    size_t total = static_cast<size_t>(N) * H * W;
    std::vector<float> hostBuf(total);
    cudaMemcpy(hostBuf.data(), burst.data(),
               total * sizeof(float), cudaMemcpyDeviceToHost);

    // 4) Compare elementwise (allow tiny epsilon)
    float maxDiff = 0.f;
    for (size_t i = 0; i < total; ++i) {
        float d = std::abs(hostBuf[i] - refData[i]);
        if (d > maxDiff) maxDiff = d;
        if (d > 1e-3f) {
            std::cerr << "Mismatch at " << i << ": got "
                      << hostBuf[i] << " vs " << refData[i] << "\n";
            return 1;
        }
    }
    std::cout << "TestBurst passed. maxDiff=" << maxDiff << "\n";
    return 0;
}