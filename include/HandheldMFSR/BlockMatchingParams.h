#pragma once
#include <vector>
#include <string>

namespace HandheldMFSR {

/// Mirrors Python’s params['block matching']['tuning']
struct BlockMatchingParams {
    std::vector<int>        factors;     // e.g. {1,2,4,4}
    std::vector<int>        tileSizes;   // same length
    std::vector<int>        searchRadia; // same length
    std::vector<std::string> distances;  // "L1" or "L2" per level
};

} // namespace HandheldMFSR
