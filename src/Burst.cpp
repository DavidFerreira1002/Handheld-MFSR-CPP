// src/Burst.cpp

#include "HandheldMFSR/Burst.h"
#include <libraw/libraw.h>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>         // for debug/logging
#include <cuda_runtime.h>

namespace fs = std::filesystem;

namespace HandheldMFSR {

Burst::Burst() = default;

Burst::~Burst() {
    if (d_data_) {
        cudaFree(d_data_);
    }
}

void Burst::loadFromDisk(const std::string& folderPath) {
    // 1) Gather RAW/DNG file paths
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".dng" || ext == ".arw" || ext == ".cr2" || ext == ".nef") {
            files.push_back(entry.path().string());
        }
    }
    if (files.empty()) {
        throw std::runtime_error("Burst::loadFromDisk: no RAW files found in " + folderPath);
    }
    std::sort(files.begin(), files.end());

    // 2) Decode first image to get dimensions
    LibRaw raw;
    if (int ret = raw.open_file(files[0].c_str())) {
        throw std::runtime_error("LibRaw failed to open " + files[0] + ": error " + std::to_string(ret));
    }
    if (int ret = raw.unpack()) {
        throw std::runtime_error("LibRaw failed to unpack " + files[0] + ": error " + std::to_string(ret));
    }
    // Raw sizes
    H_ = raw.imgdata.sizes.raw_height;
    W_ = raw.imgdata.sizes.raw_width;
    N_ = static_cast<int>(files.size());

    // 3) Allocate host buffer (N × H × W) and fill it
    size_t frameSize = static_cast<size_t>(H_) * W_;
    std::vector<float> hostBuf(static_cast<size_t>(N_) * frameSize);

    for (int i = 0; i < N_; ++i) {
        LibRaw iterRaw;
        if (int ret = iterRaw.open_file(files[i].c_str())) {
            throw std::runtime_error("LibRaw failed to open " + files[i]);
        }
        if (int ret = iterRaw.unpack()) {
            throw std::runtime_error("LibRaw failed to unpack " + files[i]);
        }
        // Pointer to 16-bit CFA data
        uint16_t* rawImg = iterRaw.imgdata.rawdata.raw_image;
        int rawPitch = iterRaw.imgdata.sizes.raw_pitch / sizeof(uint16_t);

        // Copy into hostBuf
        size_t offset = static_cast<size_t>(i) * frameSize;
        for (int y = 0; y < H_; ++y) {
            for (int x = 0; x < W_; ++x) {
                hostBuf[offset + y * W_ + x] = static_cast<float>( rawImg[y * rawPitch + x] );
            }
        }
        // Read noise-profile α,β from metadata (here we just grab the first frame’s)
        if (i == 0) {
            // LibRaw no longer exposes a `noise` member — fall back to defaults
            alpha_ = 1.0f;
            beta_  = 0.0f;
        }
        iterRaw.recycle();  // free LibRaw internals
    }

    // 4) Allocate & upload to GPU
    size_t totalBytes = hostBuf.size() * sizeof(float);
    cudaError_t cudaErr = cudaMalloc(&d_data_, totalBytes);
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(cudaErr));
    }
    cudaErr = cudaMemcpy(d_data_, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(d_data_);
        d_data_ = nullptr;
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(cudaErr));
    }

    std::cout << "[Burst] Loaded " << N_ << " frames (" << W_ << "×" << H_
              << "), alpha=" << alpha_ << ", beta=" << beta_ << "\n";
}

} // namespace HandheldMFSR