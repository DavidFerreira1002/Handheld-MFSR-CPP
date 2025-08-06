#include "HandheldMFSR/BlockMatcher.h"
#include <stdexcept>
#include <cfloat>       // for FLT_MAX
#include <cuda_runtime.h>

namespace HandheldMFSR {

//— ctor —  
BlockMatcher::BlockMatcher(BlockMatchingParams params)
  : _p(std::move(params)), _levels(int(_p.factors.size()))
{
  if (_p.tileSizes.size()   != _levels ||
      _p.searchRadia.size() != _levels ||
      _p.distances.size()   != _levels)
    throw std::invalid_argument("BlockMatchingParams vectors must have equal length");
}

// wrap into [0, M)
__device__ inline int wrap(int v, int M) {
  v %= M;
  return v < 0 ? v + M : v;
}

// -----------------------------------------------------------------------------
// 3-candidate upsample kernel — now takes both ref and cmp pointers
// -----------------------------------------------------------------------------
__global__
void upsampleKernel(
    const float*    ref,       // full-res or level-res reference image
    const float*    cmp,       // the image to align
    const int2*     prevAlign, // coarse alignments
    int2*           outAlign,  // this level’s alignments
    int             prevY, int prevX,
    int             newY,  int newX,
    int             upF, int tileSize, int prevTileSize,
    int             H,     int W)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    if (tx >= newX || ty >= newY) return;

    int idxOut = ty*newX + tx;
    int repeat = upF / (tileSize / prevTileSize);
    int px = tx / repeat, py = ty / repeat;
    if (px >= prevX || py >= prevY) {
      outAlign[idxOut] = make_int2(0,0);
      return;
    }

    // fetch base + neighbors
    int2 c0 = prevAlign[py*prevX + px];
    c0.x *= upF; c0.y *= upF;
    int sx = (2*(tx%repeat)+1 > repeat) ? 1 : -1;
    int sy = (2*(ty%repeat)+1 > repeat) ? 1 : -1;
    int nx = min(px+sx, prevX-1), ny = min(py+sy, prevY-1);

    int2 ch = prevAlign[py*prevX + nx];  ch.x *= upF; ch.y *= upF;
    int2 cv = prevAlign[ny*prevX + px];  cv.x *= upF; cv.y *= upF;

    // load reference patch into shared
    extern __shared__ float s_ref[]; // tileSize*tileSize
    int baseY = ty*tileSize, baseX = tx*tileSize;
    for (int j = threadIdx.y; j < tileSize; j += blockDim.y)
      for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
        int yy = wrap(baseY + j, H);
        int xx = wrap(baseX + i, W);
        s_ref[j*tileSize + i] = ref[yy*W + xx];
      }
    __syncthreads();

    // cost evaluation lambda
    auto evalCost = [&](int2 c) {
      float cost = 0;
      for (int j = 0; j < tileSize; ++j)
        for (int i = 0; i < tileSize; ++i) {
          int yy = wrap(baseY + j + c.y, H);
          int xx = wrap(baseX + i + c.x, W);
          float v = s_ref[j*tileSize + i] - cmp[yy*W + xx];
          cost += fabsf(v);
        }
      return cost;
    };

    // pick best of c0, ch, cv
    float bestCost = evalCost(c0);
    int2 best = c0;

    float costH = evalCost(ch);
    if (costH < bestCost) { bestCost = costH; best = ch; }

    float costV = evalCost(cv);
    if (costV < bestCost) { bestCost = costV; best = cv; }

    outAlign[idxOut] = best;
}

// — L1 local search —
__global__
void localSearchL1(
    const float* ref, const float* cmp,
    int tilesY, int tilesX, int tileSize, int rad,
    int2* align, int H, int W)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty*tilesX + tx;
    int2 f   = align[idx];

    extern __shared__ float s_ref[];
    int baseY = ty*tileSize, baseX = tx*tileSize;
    for(int j = threadIdx.y; j < tileSize; j += blockDim.y)
      for(int i = threadIdx.x; i < tileSize; i += blockDim.x) {
        int yy = wrap(baseY + j, H);
        int xx = wrap(baseX + i, W);
        s_ref[j*tileSize + i] = ref[yy*W + xx];
      }
    __syncthreads();

    float best = FLT_MAX; int2 bestF = f;
    for(int dx = -rad; dx <= rad; ++dx) {
      for(int dy = -rad; dy <= rad; ++dy) {
        float cost = 0;
        int2 c = make_int2(f.x + dx, f.y + dy);
        for(int j = 0; j < tileSize; ++j)
          for(int i = 0; i < tileSize; ++i) {
            int yy = wrap(baseY + j + c.y, H);
            int xx = wrap(baseX + i + c.x, W);
            cost += fabsf(s_ref[j*tileSize + i] - cmp[yy*W + xx]);
          }
        if (cost < best) { best = cost; bestF = c; }
      }
    }
    align[idx] = bestF;
}

// — L2 local search —
__global__
void localSearchL2(
    const float* ref, const float* cmp,
    int tilesY, int tilesX, int tileSize, int rad,
    int2* align, int H, int W)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int idx = ty*tilesX + tx;
    int2 f   = align[idx];

    extern __shared__ float s_ref[];
    int baseY = ty*tileSize, baseX = tx*tileSize;
    for(int j = threadIdx.y; j < tileSize; j += blockDim.y)
      for(int i = threadIdx.x; i < tileSize; i += blockDim.x) {
        int yy = wrap(baseY + j, H);
        int xx = wrap(baseX + i, W);
        s_ref[j*tileSize + i] = ref[yy*W + xx];
      }
    __syncthreads();

    float best = FLT_MAX; int2 bestF = f;
    for(int dx = -rad; dx <= rad; ++dx) {
      for(int dy = -rad; dy <= rad; ++dy) {
        float cost = 0;
        int2 c = make_int2(f.x + dx, f.y + dy);
        for(int j = 0; j < tileSize; ++j)
          for(int i = 0; i < tileSize; ++i) {
            int yy = wrap(baseY + j + c.y, H);
            int xx = wrap(baseX + i + c.x, W);
            float d = s_ref[j*tileSize + i] - cmp[yy*W + xx];
            cost += d*d;
          }
        if (cost < best) { best = cost; bestF = c; }
      }
    }
    align[idx] = bestF;
}

//— host match() —  
__host__
thrust::device_vector<int2> BlockMatcher::match(
    const std::vector<thrust::device_vector<float>>&            refPyr,
    const std::vector<std::vector<thrust::device_vector<float>>>& compPyrs,
    int H0, int W0)
{
  int N = int(compPyrs.size()), L = _levels;
  thrust::device_vector<int2> alignAll, alignPrev, alignCurr;

  for(int lv = 0; lv < L; ++lv) {
  int rev = L - 1 - lv;  // “reverse index”: lv=0→last level (coarsest), etc.

  int f      = _p.factors   [rev];  // Python’s factors[lv]
  int Ts     = _p.tileSizes [rev];
  int rad    = _p.searchRadia[rev];
  bool useL2 = (_p.distances[rev] == "L2");

  int H = H0 / f, W = W0 / f;
  int tilesY = (H + Ts - 1) / Ts;
  int tilesX = (W + Ts - 1) / Ts;
  size_t M   = size_t(N)*tilesY*tilesX;

  alignCurr.resize(M);

  if (lv == 0) {
    // coarsest → zero
    cudaMemset(alignCurr.data().get(), 0, M*sizeof(int2));
  } else {
    int prevF     = _p.factors   [lv-1];
    int prevTs    = _p.tileSizes [lv-1];
    int prevTilesY= (H0/prevF + prevTs - 1)/prevTs;
    int prevTilesX= (W0/prevF + prevTs - 1)/prevTs;
    int upF       = prevF / f; 

    dim3 tb(16,16), bg((tilesX+15)/16,(tilesY+15)/16);
    for(int i=0;i<N;++i){
      const float* refLvl = refPyr[rev].data().get();
      const float* cmpLvl = compPyrs[i][rev].data().get();
      const int2* in      = alignPrev.data().get() + i*prevTilesY*prevTilesX;
      int2*       out     = alignCurr.data().get() + i*tilesY*tilesX;

      upsampleKernel<<<bg,tb,Ts*Ts*sizeof(float)>>>(
          refLvl, cmpLvl,
          in, out,
          prevTilesY, prevTilesX,
          tilesY,    tilesX,
          upF,      
          Ts, prevTs,
          H, W);
    }
  }

    // local search (unchanged)
    {
      dim3 tb(16,16), bg((tilesX+15)/16, (tilesY+15)/16);
      size_t shared = Ts*Ts*sizeof(float);
      for(int i=0; i<N; ++i) {
        const float* refLvl = refPyr[rev].data().get();
        const float* cmpLvl = compPyrs[i][rev].data().get();
        int2*        ptr    = alignCurr.data().get() + i*tilesY*tilesX;

        if (useL2)
          localSearchL2<<<bg,tb,shared>>>(refLvl, cmpLvl, tilesY, tilesX, Ts, rad, ptr, H, W);
        else
          localSearchL1<<<bg,tb,shared>>>(refLvl, cmpLvl, tilesY, tilesX, Ts, rad, ptr, H, W);
      }
    }

    // append
    if (lv == 0) {
      alignAll = alignCurr;
    } else {
      auto old = alignAll.size();
      alignAll.resize(old + alignCurr.size());
      cudaMemcpy(
        alignAll.data().get() + old,
        alignCurr.data().get(),
        alignCurr.size() * sizeof(int2),
        cudaMemcpyDeviceToDevice);
    }
    alignPrev.swap(alignCurr);
  }

  return alignAll;
}

} // namespace HandheldMFSR
