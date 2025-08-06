#include "HandheldMFSR/GreyConverter.h"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace HandheldMFSR {

// Shared thread block size for decimation
static constexpr int DEC_X = 16, DEC_Y = 16;

// Kernel: decimate 2×2 → average
__global__ void decimateKernel(const float* in, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = W/2, outH = H/2;
    if (x < outW && y < outH) {
        int ix = 2*x, iy = 2*y;
        float sum = in[iy*W + ix]
                  + in[iy*W + ix+1]
                  + in[(iy+1)*W + ix]
                  + in[(iy+1)*W + ix+1];
        out[y*outW + x] = sum * 0.25f;
    }
}

// Kernel: pack real→complex
__global__ void packComplex(const float* in, cufftComplex* out, int H, int W) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = H*W;
    if (idx < total) {
        out[idx].x = in[idx];
        out[idx].y = 0.0f;
    }
}

__global__ void zeroHighFreq(cufftComplex* data, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;
    int u = idx / W;  // frequency row
    int v = idx % W;  // frequency col

    int h4 = H / 4;
    int w4 = W / 4;
    bool low_u = (u <= h4) || (u >= H - h4);
    bool low_v = (v <= w4) || (v >= W - w4);

    if (!(low_u && low_v)) {
        data[idx].x = 0.0f;
        data[idx].y = 0.0f;
    }
}
// Kernel: unpack complex→real and apply scale factor
__global__ void unpackAndScale(const cufftComplex* in,
                              float* out, int H, int W,
                              float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx < total) {
        out[idx] = in[idx].x * scale;
    }
}

thrust::device_vector<float> computeGrey(
    const float* d_in, int H, int W, GreyMethod method)
{
    if (method == GreyMethod::Decimate) {
        int outH = H/2, outW = W/2;
        thrust::device_vector<float> d_out(size_t(outH)*outW);
        dim3 threads(DEC_X, DEC_Y);
        dim3 blocks((outW + DEC_X -1)/DEC_X, (outH + DEC_Y -1)/DEC_Y);
        decimateKernel<<<blocks,threads>>>(d_in, d_out.data().get(), H, W);
        cudaDeviceSynchronize();
        return d_out;
    }

    // --- FFT path ---
    int N = H*W;
    // 1) pack into complex array
    cufftComplex* d_freq;
    cudaMalloc(&d_freq, sizeof(cufftComplex)*N);
    {
        int block = 256, grid = (N+block-1)/block;
        packComplex<<<grid,block>>>(d_in, d_freq, H, W);
        cudaDeviceSynchronize();
    }

    // 2) forward FFT (in-place C2C)
    cufftHandle plan;
    cufftPlan2d(&plan, H, W, CUFFT_C2C);
    cufftExecC2C(plan, d_freq, d_freq, CUFFT_FORWARD);

    {
    int block = 256, grid = (N + block - 1) / block;
    zeroHighFreq<<<grid,block>>>(d_freq, H, W);
    cudaDeviceSynchronize();
    }

    // 4) inverse FFT
    cufftExecC2C(plan, d_freq, d_freq, CUFFT_INVERSE);
    cufftDestroy(plan);

    // 5) unpack real part and normalize by (H*W) due to CUFFT scaling
    float scale = 1.0f / float(H * W);
    thrust::device_vector<float> d_out(N);
    {
        int block = 256, grid = (N+block-1)/block;
        unpackAndScale<<<grid, block>>>(d_freq, d_out.data().get(), H, W, scale);
        cudaDeviceSynchronize();
    }
    cudaFree(d_freq);
    return d_out;
}

} // namespace HandheldMFSR
