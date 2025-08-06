#pragma once
#include "BlockMatchingParams.h"
#include <vector>
#include <thrust/device_vector.h>
#include <vector_types.h>   // for int2

namespace HandheldMFSR {

/// Performs the multi‐level block‐matching exactly as in
/// registration/block_matching.py (HDR+ three-candidate upsample + L1/L2 search).
class BlockMatcher {
public:
  explicit BlockMatcher(BlockMatchingParams params);

  /// refPyr:   coarse→fine pyramid of the reference (levels L)
  /// compPyrs: for each of N bursts, the same pyramid structure
  /// H0,W0:    full-res dims
  /// returns N×L×tilesY×tilesX int2 shifts, flattened
  __host__
  thrust::device_vector<int2> match(
    const std::vector<thrust::device_vector<float>>&            refPyr,
    const std::vector<std::vector<thrust::device_vector<float>>>& compPyrs,
    int H0, int W0);

private:
  BlockMatchingParams  _p;
  int                  _levels;
};

} // namespace HandheldMFSR
