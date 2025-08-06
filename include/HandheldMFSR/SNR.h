#pragma once
#include "HandheldMFSR/NoiseModel.h"
#include "HandheldMFSR/RunOptions.h"   // for PipelineParams
#include <cuda_runtime.h>
#include "RunOptions.h" // for PipelineParams
#include <vector>
#include <string>

namespace HandheldMFSR {

/// Copy the device reference frame (H×W floats) to host and compute mean brightness.
float computeBrightness(const float* d_ref, int H, int W);

/// Given mean brightness and a precomputed noise‐model, estimate SNR.
float estimateSNR(float brightness, const NoiseModel& nm);

/// Based on SNR, returns a PipelineParams instance tuned via Python’s get_params(SNR).
/// For now this can be a stub mapping rough SNR bands to param presets.
PipelineParams getParamsForSNR(float SNR);

} // namespace HandheldMFSR
