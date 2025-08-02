#!/usr/bin/env python3
"""
extract_test_burst.py

Loads the first N frames from a DNG burst folder and saves them
as a NumPy array for C++ unit-testing (shape: N×H×W, dtype=float32).
"""

import os
import sys
import argparse

import rawpy
import numpy as np

def load_burst(burst_folder, max_frames=None):
    # gather DNGs
    files = sorted([
        os.path.join(burst_folder, f)
        for f in os.listdir(burst_folder)
        if f.lower().endswith('.dng')
    ])
    if not files:
        raise RuntimeError(f"No .dng files in {burst_folder}")
    if max_frames:
        files = files[:max_frames]
    # decode
    frames = []
    for p in files:
        with rawpy.imread(p) as raw:
            # get the raw mosaic as uint16
            raw_img = raw.raw_image.copy()
            frames.append(raw_img.astype(np.float32))
    return np.stack(frames, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('burst_folder', help="Folder containing .dng frames")
    parser.add_argument('--out', default='test_burst.npy', help="Output .npy file")
    args = parser.parse_args()

    arr = load_burst(args.burst_folder)
    # arr.shape = (N, H, W)
    np.save(args.out, arr)
    print(f"Saved {arr.shape} array to {args.out}")

if __name__ == '__main__':
    main()