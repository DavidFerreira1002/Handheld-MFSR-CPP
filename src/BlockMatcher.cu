// src/BlockMatcher.cu
#include "HandheldMFSR/BlockMatcher.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

namespace HandheldMFSR {

//–– out-of-line ctor definition ––
BlockMatcher::BlockMatcher(const BlockMatchingParams& p)
  : params(p),
    levels(int(p.factors.size()))
{
  if (p.tileSizes.size() != levels ||
      p.searchRadia.size() != levels ||
      p.distances.size() != levels)
    throw std::invalid_argument("BlockMatchingParams vectors must all have same length");
}

namespace {

/// Circular wrap for coordinates
__device__ inline int wrap(int v, int M) {
  v %= M;
  return (v < 0) ? v + M : v;
}

// -----------------------------------------------------------------------------
// 3-candidate upsample kernel (temporarily replaced by a straight copy)
// -----------------------------------------------------------------------------
__global__
void upsampleKernel(const int2* prevAlign, int2* outAlign,
                    int prevY, int prevX,
                    int newY, int newX,
                    int /*upF*/, int /*tileSize*/, int /*prevTileSize*/,
                    int /*H*/, int /*W*/)
{
  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = blockIdx.y*blockDim.y + threadIdx.y;
  if (tx >= newX || ty >= newY) return;

  // Simple: copy from previous (clamp if out-of-bounds)
  int px = min(tx, prevX-1);
  int py = min(ty, prevY-1);
  outAlign[ty*newX + tx] = prevAlign[py*prevX + px];
}

// -----------------------------------------------------------------------------
// Local search kernel (L1)
// -----------------------------------------------------------------------------
__global__
void localSearchL1(const float* ref, const float* cmp,
                   int tilesY, int tilesX,
                   int tileSize, int rad,
                   int2* align,
                   int H, int W)
{
  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = blockIdx.y*blockDim.y + threadIdx.y;
  if (tx>=tilesX || ty>=tilesY) return;

  int idx = ty*tilesX + tx;
  int2 f = align[idx];

  extern __shared__ float s_ref[];
  int baseY = ty*tileSize, baseX = tx*tileSize;

  // load ref patch
  for(int j=threadIdx.y; j<tileSize; j+=blockDim.y)
    for(int i=threadIdx.x; i<tileSize; i+=blockDim.x) {
      int yy = wrap(baseY + j, H), xx = wrap(baseX + i, W);
      s_ref[j*tileSize + i] = ref[yy*W + xx];
    }
  __syncthreads();

  float best = FLT_MAX;
  int2 bestF = f;

  for(int dy=-rad; dy<=rad; ++dy) {
    for(int dx=-rad; dx<=rad; ++dx) {
      float cost = 0;
      int2 c{ f.x + dx, f.y + dy };
      for(int j=0; j<tileSize; ++j) {
        for(int i=0; i<tileSize; ++i) {
          int yy = wrap(baseY + j + c.y, H),
              xx = wrap(baseX + i + c.x, W);
          cost += fabsf(s_ref[j*tileSize + i] - cmp[yy*W + xx]);
        }
      }
      if (cost < best) {
        best = cost;
        bestF = c;
      }
    }
  }
  align[idx] = bestF;
}

// -----------------------------------------------------------------------------
// Local search kernel (L2)
// -----------------------------------------------------------------------------
__global__
void localSearchL2(const float* ref, const float* cmp,
                   int tilesY, int tilesX,
                   int tileSize, int rad,
                   int2* align,
                   int H, int W)
{
  int tx = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = blockIdx.y*blockDim.y + threadIdx.y;
  if (tx>=tilesX || ty>=tilesY) return;

  int idx = ty*tilesX + tx;
  int2 f = align[idx];

  extern __shared__ float s_ref[];
  int baseY = ty*tileSize, baseX = tx*tileSize;

  // load ref patch
  for(int j=threadIdx.y; j<tileSize; j+=blockDim.y)
    for(int i=threadIdx.x; i<tileSize; i+=blockDim.x) {
      int yy = wrap(baseY + j, H), xx = wrap(baseX + i, W);
      s_ref[j*tileSize + i] = ref[yy*W + xx];
    }
  __syncthreads();

  float best = FLT_MAX;
  int2 bestF = f;

  for(int dy=-rad; dy<=rad; ++dy) {
    for(int dx=-rad; dx<=rad; ++dx) {
      float cost = 0;
      int2 c{ f.x + dx, f.y + dy };
      for(int j=0; j<tileSize; ++j) {
        for(int i=0; i<tileSize; ++i) {
          int yy = wrap(baseY + j + c.y, H),
              xx = wrap(baseX + i + c.x, W);
          float d = s_ref[j*tileSize + i] - cmp[yy*W + xx];
          cost += d*d;
        }
      }
      if (cost < best) {
        best = cost;
        bestF = c;
      }
    }
  }
  align[idx] = bestF;
}

} // anonymous namespace


//------------------------------------------------------------------------------
// Host implementation of match()
//------------------------------------------------------------------------------
thrust::device_vector<int2> BlockMatcher::match(
    const std::vector<thrust::device_vector<float>>& refPyr,
    const std::vector<std::vector<thrust::device_vector<float>>>& compPyrs,
    int H0, int W0)
{
  int N = int(compPyrs.size());
  int L = levels;

  thrust::device_vector<int2> alignAll;
  thrust::device_vector<int2> alignPrev, alignCurr;

  for(int lv=0; lv<L; ++lv) {
    int f     = params.factors[lv];
    int H     = H0 / f, W = W0 / f;
    int Ts    = params.tileSizes[lv];
    int rad   = params.searchRadia[lv];
    bool useL2= (params.distances[lv] == "L2");

    int tilesY = (H + Ts -1)/Ts;
    int tilesX = (W + Ts -1)/Ts;
    size_t M   = size_t(N) * tilesY * tilesX;

    // allocate current-level alignment
    alignCurr.resize(M);

    // initial or upsample
    if (lv == 0) {
      cudaMemset(alignCurr.data().get(), 0, M * sizeof(int2));
    } else {
      // copy previous directly
      cudaMemcpy(alignCurr.data().get(),
                 alignPrev.data().get(),
                 M * sizeof(int2),
                 cudaMemcpyDeviceToDevice);
    }

    // local search per image
    for(int i=0; i<N; ++i){
      const float* refLvl = refPyr[lv].data().get();
      const float* cmpLvl = compPyrs[i][lv].data().get();
      int2* alignPtr      = alignCurr.data().get() + i*tilesY*tilesX;
      dim3 tb(16,16),
           bg((tilesX+15)/16,(tilesY+15)/16);
      size_t shared = Ts*Ts * sizeof(float);

      if (useL2) {
        localSearchL2<<<bg,tb,shared>>>(refLvl,cmpLvl,tilesY,tilesX,Ts,rad,alignPtr,H,W);
      } else {
        localSearchL1<<<bg,tb,shared>>>(refLvl,cmpLvl,tilesY,tilesX,Ts,rad,alignPtr,H,W);
      }
    }

    // accumulate into alignAll
    if (lv == 0) {
      alignAll = alignCurr;
    } else {
      auto old = alignAll.size();
      alignAll.resize(old + alignCurr.size());
      cudaMemcpy(alignAll.data().get()+old,
                 alignCurr.data().get(),
                 alignCurr.size()*sizeof(int2),
                 cudaMemcpyDeviceToDevice);
    }

    alignPrev.swap(alignCurr);
  }

  return alignAll;
}

} // namespace HandheldMFSR
