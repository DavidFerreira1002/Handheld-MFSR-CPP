# Handheld Multi-Frame Super-Resolution in C++/CUDA

Port of https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution  
Original (Python) → This (C++/CUDA)

## Project Structure

- `src/`: C++ source files  
- `include/`: Public headers  
- `build/`: Out-of-source CMake builds  
- `scripts/`: Data-preparation and test scripts  

## Requirements

- CMake ≥ 3.18  
- CUDA Toolkit ≥ 11.0  
- OpenCV ≥ 4.5  

## Build & Run

```bash
mkdir build && cd build
cmake .. && make -j
./HandheldMFSR [args…]