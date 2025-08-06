#include "HandheldMFSR/Burst.h"
#include <libraw/libraw.h>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

namespace fs = std::filesystem;
using namespace HandheldMFSR;

Burst::Burst() = default;

Burst::~Burst() {
    if (d_refData_)  cudaFree(d_refData_);
    if (d_compData_) cudaFree(d_compData_);
}

void Burst::loadFromDisk(const std::string& folderPath) {
    // 1) Enumerate RAW files
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(folderPath)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext==".dng"||ext==".arw"||ext==".cr2"||ext==".nef")
            files.push_back(e.path().string());
    }
    if (files.empty())
        throw std::runtime_error("No RAW files found in " + folderPath);
    std::sort(files.begin(), files.end());
    N_ = static_cast<int>(files.size());

    // 2) Decode reference frame & extract metadata
    LibRaw raw;
    if (int ret = raw.open_file(files[0].c_str()))
        throw std::runtime_error("LibRaw open failed: " + std::to_string(ret));
    if (int ret = raw.unpack())
        throw std::runtime_error("LibRaw unpack failed: " + std::to_string(ret));

    H_ = raw.imgdata.sizes.raw_height;
    W_ = raw.imgdata.sizes.raw_width;

    // Extract tags
    auto& T = tags_;
    // ISO
    int iso = raw.imgdata.other.iso_speed;
    T.ISO = std::clamp(iso, 100, 3200);
    // White & black levels
    T.white_level = raw.imgdata.color.maximum; 
    // LibRaw uses one black level, but DNG can have per-channel:
    // we'll duplicate for all four Bayer channels
    int black = raw.imgdata.color.black;
    T.black_levels = {black, black, black, black}; 
    // White-balance multipliers
    auto& cam_mul = raw.imgdata.color.cam_mul;
    T.white_balance = {cam_mul[0], cam_mul[1], cam_mul[2], cam_mul[1]}; 
    // CFA pattern (2×2)
    // Most DNG bursts use RGGB. We default to that in BurstTags.
    // If you need to override from metadata, you can parse raw.imgdata.sizes.filters or EXIF later.
    // XYZ→camera 3×3
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j)
            T.xyz2cam[i][j] = raw.imgdata.color.rgb_cam[i][j];

    // Allocate host ref buffer
    size_t frameSize = size_t(H_)*W_;
    std::vector<float> refBuf(frameSize);

    // Copy raw 16→float
    {
        auto ptr16 = raw.imgdata.rawdata.raw_image;
        int pitch = raw.imgdata.sizes.raw_pitch/sizeof(uint16_t);
        for (int y=0; y<H_; ++y)
            for (int x=0; x<W_; ++x)
                refBuf[y*W_ + x] = float(ptr16[y*pitch + x]);
    }
    raw.recycle();  // free LibRaw internals

    // 3) Decode comp frames into one host buffer
    std::vector<float> compBuf(size_t(N_)*frameSize);
    for (int i=1; i<N_; ++i) {
        LibRaw R;
        if (int ret = R.open_file(files[i].c_str()))
            throw std::runtime_error("LibRaw open failed: " + files[i]);
        if (int ret = R.unpack())
            throw std::runtime_error("LibRaw unpack failed: " + files[i]);

        auto ptr16 = R.imgdata.rawdata.raw_image;
        int pitch = R.imgdata.sizes.raw_pitch/sizeof(uint16_t);
        size_t offset = size_t(i)*frameSize;
        for (int y=0; y<H_; ++y) {
            for (int x=0; x<W_; ++x) {
                compBuf[offset + y*W_ + x] = float(ptr16[y*pitch + x]);
            }
        }
        R.recycle();
    }

    // 4) Host-side normalization (black/white-level + white-balance)
    auto normalize = [&](float* data, size_t count){
        for (size_t idx = 0; idx < count; ++idx) {
            int y = idx / W_;
            int x = idx % W_;
            int c = T.CFA[(y&1)][(x&1)];       // channel 0..2
            float bl = T.black_levels[c];
            float wl = float(T.white_level - T.black_levels[c]);
            float wb = T.white_balance[c] / T.white_balance[1];
            float v = (data[idx] - bl) / wl;
            data[idx] = std::clamp(v * wb, 0.0f, 1.0f);
        }
    };
    normalize(refBuf.data(), frameSize);
    normalize(compBuf.data(), compBuf.size());

    // 5) cudaMalloc + cudaMemcpy to device
    cudaError_t err;
    err = cudaMalloc(&d_refData_, frameSize * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMalloc(&d_compData_, compBuf.size() * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    err = cudaMemcpy(d_refData_, refBuf.data(),
                     frameSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMemcpy(d_compData_, compBuf.data(),
                     compBuf.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    std::cout << "[Burst] Loaded " << N_ << " frames of size "
              << W_ << "×" << H_ << ", ISO=" << T.ISO << "\n";
}
