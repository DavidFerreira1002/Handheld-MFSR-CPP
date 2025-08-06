// src/Pyramid.cu

#include "HandheldMFSR/Pyramid.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <algorithm>            // for std::reverse
#include <stdexcept>

namespace HandheldMFSR {

// Reflect‐padding helper
__device__ inline int reflect(int x, int N) {
    if (x < 0)      return -x - 1;
    if (x >= N)     return 2*N - x - 1;
    return x;
}

// Horizontal convolution
__global__ void convH(const float* in, float* out,
                      int H, int W,
                      const float* ker, int kRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sum = 0.f;
    for (int i = -kRadius; i <= kRadius; ++i) {
        int xx = reflect(x + i, W);
        sum += ker[i + kRadius] * in[y*W + xx];
    }
    out[y*W + x] = sum;
}

// Vertical convolution
__global__ void convV(const float* in, float* out,
                      int H, int W,
                      const float* ker, int kRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sum = 0.f;
    for (int i = -kRadius; i <= kRadius; ++i) {
        int yy = reflect(y + i, H);
        sum += ker[i + kRadius] * in[yy*W + x];
    }
    out[y*W + x] = sum;
}

// Downsample by integer factor (simple strided copy)
__global__ void downsampleKernel(const float* in, float* out,
                                 int H, int W, int factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = W / factor;
    int outH = H / factor;
    if (x < outW && y < outH) {
        out[y*outW + x] = in[(y*factor)*W + (x*factor)];
    }
}

std::vector<thrust::device_vector<float>>
buildGaussianPyramid(const float* d_in, int H, int W,
                     const PyramidParams& params)
{
    if (params.factors.empty())
        throw std::invalid_argument("PyramidParams.factors must be non-empty");

    // Curr holds the current level, starting at input
    thrust::device_vector<float> curr(d_in, d_in + size_t(H)*W);
    // Temp for separable conv
    thrust::device_vector<float> temp(size_t(H)*W);

    std::vector<thrust::device_vector<float>> levels;
    dim3 threads(16,16);
    dim3 blocks((W+15)/16, (H+15)/16);

    for (int f : params.factors) {
        // 1) Gaussian blur
        float sigma = f * params.sigmaScale;
        auto ker = gaussianKernel1D(sigma);
        int kRadius = (int(ker.size()) - 1)/2;
        thrust::device_vector<float> d_ker = ker;

        // Horizontal pass
        convH<<<blocks,threads>>>(curr.data().get(), temp.data().get(),
                                  H, W,
                                  d_ker.data().get(), kRadius);
        cudaDeviceSynchronize();

        // Vertical pass
        convV<<<blocks,threads>>>(temp.data().get(), curr.data().get(),
                                  H, W,
                                  d_ker.data().get(), kRadius);
        cudaDeviceSynchronize();

        // 2) Downsample
        int outH = H / f, outW = W / f;
        thrust::device_vector<float> down(size_t(outH)*outW);
        dim3 dBlocks((outW+15)/16, (outH+15)/16);
        downsampleKernel<<<dBlocks,threads>>>(curr.data().get(),
                                               down.data().get(),
                                               H, W, f);
        cudaDeviceSynchronize();

        levels.push_back(std::move(down));
    }

    // Reverse to coarse→fine
    std::reverse(levels.begin(), levels.end());
    return levels;
}

} // namespace HandheldMFSR
