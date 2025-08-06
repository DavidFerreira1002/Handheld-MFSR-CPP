#pragma once
#include <vector>
#include <vector_types.h> 
#include <string>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

namespace HandheldMFSR {

/// Exactly mirroring Python’s params['block matching']['tuning']
struct BlockMatchingParams {
    std::vector<int>       factors;      // [1,2,4,4]
    std::vector<int>       tileSizes;    // Ts per level
    std::vector<int>       searchRadia;  // radius per level
    std::vector<std::string> distances;  // "L1" or "L2" per level
};

/// BlockMatcher carries the tuning params and runs per‐level, per‐tile
/// integer alignment on the GPU.
class BlockMatcher {
public:
  BlockMatcher(const BlockMatchingParams& p);
  /// refPyr: coarse→fine pyramid for reference (levels = L)
  /// compPyrs: for each of N images, a pyramid same structure
  /// H0,W0: full‐res dims
  /// Returns a flat dev‐vector of N*L*tilesY*tilesX Int2’s:
  ///   index = (((i*L + lv)*tilesY + ty)*tilesX + tx)
  thrust::device_vector<int2> match(
    const std::vector<thrust::device_vector<float>>& refPyr,
    const std::vector<std::vector<thrust::device_vector<float>>>& compPyrs,
    int H0, int W0) __host__;

private:
  BlockMatchingParams params;
  int levels;

  // GPU kernels (defined in .cu)
  void upsampleAlignmentsKernel(int2* d_align_in,
                                 int2* d_align_out,
                                 int prevTilesY, int prevTilesX,
                                 int newTilesY, int newTilesX,
                                 int upFactor, int tileSize, int prevTileSize,
                                 int H, int W, cudaStream_t);

  void localSearchKernel(const float* d_ref, const float* d_comp,
                         int tilesY, int tilesX,
                         int tileSize, int searchRad,
                         int2* d_align, bool useL2,
                         int H, int W, cudaStream_t);
};

} // namespace HandheldMFSR
