#include "HandheldMFSR/BlockMatcher.h"
#include "HandheldMFSR/Pyramid.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

// build a simple 8×8 checkerboard: value = (x+y)%2
static std::vector<float> makeChecker(int H, int W) {
    std::vector<float> v(H*W);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x)
        v[y*W+x] = float((x+y)&1);
    return v;
}

// shift image circularly by (dx,dy)
static std::vector<float> shift(const std::vector<float>& in, int H, int W, int dx, int dy) {
    std::vector<float> out(H*W);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        int xs = (x - dx + W)%W;
        int ys = (y - dy + H)%H;
        out[y*W+x] = in[ys*W+xs];
    }
    return out;
}

int main(){
    using namespace HandheldMFSR;
    const int H=8, W=8;
    auto refHost = makeChecker(H,W);
    auto compHost= shift(refHost,H,W,1,-1);

    // upload
    thrust::device_vector<float> d_ref(refHost);
    thrust::device_vector<float> d_comp(compHost);

    // build pyramids
    PyramidParams pp{{1,2,4},0.5f};
    auto refP = buildGaussianPyramid(d_ref.data().get(),H,W,pp);
    std::vector<std::vector<thrust::device_vector<float>>> comps(1);
    comps[0] = buildGaussianPyramid(d_comp.data().get(),H,W,pp);

    // BlockMatcher params
    BlockMatchingParams bmp{{1,2,4},{2,2,2},{1,1,1},{"L1","L1","L1"}};
    BlockMatcher bm(bmp);
    thrust::device_vector<int2> shifts = bm.match(refP, comps, H, W);

    // Extract finest level shifts: it's the last N*tiles*tiles entries
    int lv = 2;
    int Ts = bmp.tileSizes[lv];
    int tilesY = (H/ bmp.factors[lv] + Ts -1)/Ts;
    int tilesX = (W/ bmp.factors[lv] + Ts -1)/Ts;
    size_t offset = size_t(0)*3*tilesY*tilesX +      // image 0
                    size_t(0)*tilesY*tilesX;          // levels 0,1 skipped
    // copy back to host
    thrust::host_vector<int2> hshifts = shifts;
    for(int ty=0; ty<tilesY; ++ty){
      for(int tx=0; tx<tilesX; ++tx){
        int2 s = hshifts[offset + ty*tilesX + tx];
        assert(s.x == 1 && s.y == -1);
      }
    }
    std::cout<<"TestBlockMatcher passed\n";
    return 0;
}
