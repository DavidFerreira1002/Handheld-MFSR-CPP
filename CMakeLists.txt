cmake_minimum_required(VERSION 3.18)
project(HandheldMFSR_CUDA LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Locate LibRaw via pkg-config
pkg_check_modules(LIBRAW REQUIRED libraw)

# cnpy manual setup
set(CNPY_INCLUDE_DIR /usr/local/include)
set(CNPY_LIBRARY /usr/local/lib/libcnpy.a)

add_library(handheld_lib
    src/Burst.cpp
    src/NoiseModel.cpp
    src/SNR.cpp
    src/GreyConverter.cu
    src/GaussianKernel1D.cpp
    src/Pyramid.cu
    src/BlockMatcher.cu
    # Add any other .cpp files you want shared between main and tests
)

# Tell CMake to compile .cu with separable compilation (so symbols get emitted)
set_target_properties(handheld_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


target_include_directories(handheld_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${LIBRAW_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(handheld_lib
  PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cufft      # in case any CUFFT calls live in the library
    ${LIBRAW_LIBRARIES}
)

# Add the executable target
add_executable(HandheldMFSR src/main.cpp)
target_link_libraries(HandheldMFSR PRIVATE handheld_lib)

# Include directories
target_include_directories(HandheldMFSR PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${LIBRAW_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIR}
)

# Compiler flags from pkg-config (if any)
target_compile_options(HandheldMFSR PRIVATE
    ${LIBRAW_CFLAGS_OTHER}
)

# Link libraries
target_link_libraries(HandheldMFSR PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${LIBRAW_LIBRARIES}
    ${CNPY_LIBRARY}
)

# Link flags from pkg-config (if any)
target_link_options(HandheldMFSR PRIVATE
    ${LIBRAW_LDFLAGS_OTHER}
)

# Unit tests option
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
