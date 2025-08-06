#include "HandheldMFSR/Pyramid.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace HandheldMFSR {

// simple 2×2 decimate kernel: out[y,x] = in[(y*factor),(x*factor)]
__global__ static void decimateKernel(
    const float* __restrict__ in,
    int W0, int H0,
    float* __restrict__ out,
    int w, int h,
    int factor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int srcY = y * factor;
  int srcX = x * factor;
  // wrap or clamp?  Original code pads circularly, so:
  srcY = (srcY % H0 + H0) % H0;
  srcX = (srcX % W0 + W0) % W0;
  out[y*w + x] = in[srcY*W0 + srcX];
}

std::vector<thrust::device_vector<float>>
buildDecimationPyramid(const float* d_in,
                       int H0, int W0,
                       const std::vector<int>& factors)
{
  if (H0 <= 0 || W0 <= 0) throw std::invalid_argument("Invalid image size");
  std::vector<thrust::device_vector<float>> pyramid;
  dim3 tb(16,16);
  for (int f : factors) {
    if (f <= 0) throw std::invalid_argument("Invalid decimation factor");
    int h = (H0 + f-1)/f;
    int w = (W0 + f-1)/f;
    thrust::device_vector<float> level(w * h);
    float* d_out = level.data().get();
    dim3 bg((w + tb.x-1)/tb.x, (h + tb.y-1)/tb.y);
    decimateKernel<<<bg,tb>>>(
        d_in, W0, H0,
        d_out, w, h,
        f);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("decimateKernel launch failed: ")
                               + cudaGetErrorString(err));
    }
    pyramid.push_back(std::move(level));
  }
  return pyramid;
}

} // namespace HandheldMFSR
