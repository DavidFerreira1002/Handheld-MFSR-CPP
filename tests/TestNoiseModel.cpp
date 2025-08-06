#include "HandheldMFSR/NoiseModel.h"
#include <cnpy.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    using namespace HandheldMFSR;

    // 1) Load Python reference arrays
    cnpy::NpyArray stdArr  = cnpy::npy_load("tests/data/noise_std.npy");
    cnpy::NpyArray diffArr = cnpy::npy_load("tests/data/noise_diff.npy");
    assert(stdArr.shape.size()  == 1);
    assert(diffArr.shape.size() == 1);
    int N = static_cast<int>(stdArr.shape[0]);
    assert(diffArr.shape[0] == N);

    // Pointer to the double data
    const double* stdRefD  = stdArr.data<double>();
    const double* diffRefD = diffArr.data<double>();

    // Convert to float for comparison
    std::vector<float> stdRef (N), diffRef(N);
    for (int i = 0; i < N; ++i) {
        stdRef [i] = static_cast<float>(stdRefD [i]);
        diffRef[i] = static_cast<float>(diffRefD[i]);
    }

    // 2) Run our C++ Monte Carlo port
    float alpha = /* same α as Python */ 1.80710882e-4f;
    float beta  = /* same β as Python */ 3.1937599182128e-6f;
    NoiseModel nm = MonteCarlo::compute(alpha, beta, N-1);
    assert(static_cast<int>(nm.stdCurve.size())  == N);
    assert(static_cast<int>(nm.diffCurve.size()) == N);

    // 3) Compare element‐wise with a tolerance
    const float eps = 1e-4f;
    for (int i = 0; i < N; ++i) {
        float d0 = std::fabs(nm.stdCurve [i] - stdRef[i]);
        float d1 = std::fabs(nm.diffCurve[i] - diffRef[i]);
        if (d0 > eps) {
            std::cerr << "stdCurve mismatch at " << i 
                      << ": got " << nm.stdCurve[i]
                      << " vs " << stdRef[i] << "\n";
            return 1;
        }
        if (d1 > eps) {
            std::cerr << "diffCurve mismatch at " << i 
                      << ": got " << nm.diffCurve[i]
                      << " vs " << diffRef[i] << "\n";
            return 2;
        }
    }

    std::cout << "TestNoiseModel passed (N=" << N << ", eps=" << eps << ")\n";
    return 0;
}
