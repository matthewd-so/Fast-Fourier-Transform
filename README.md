
# FFT Implementation: CPU and GPU

## Overview

This repository contains both **CPU** and **GPU** implementations of the **Fast Fourier Transform (FFT)** in **C** and **CUDA**. You can run the FFT on a Mac/Linux machine without a CUDA-capable GPU (CPU version) or on a system with an NVIDIA GPU (GPU version).

FFT transforms a time-domain signal into its frequency-domain representation, revealing how much of each frequency is present in the original signal. This is fundamental in signal processing, communications, audio analysis, and many areas of science and engineering.

---

## Mathematical Background

### Discrete Fourier Transform (DFT)

Given a sequence of \(N\) complex samples \(x[0], x[1], \dots, x[N-1]\), the DFT is defined as:

\[
X[k] \;=\; \sum_{n=0}^{N-1} x[n] \, e^{-\,j \frac{2\pi}{N} k n}
\quad \text{for} \quad k = 0, 1, \dots, N-1.
\]

- \(X[k]\) is a complex number representing the amplitude and phase of the frequency component at \(k \times \frac{f_s}{N}\), where \(f_s\) is the sampling rate.
- Direct computation of the DFT requires \(O(N^2)\) operations.

### Fast Fourier Transform (FFT)

The FFT is an algorithm that computes the same DFT result in **\(O(N \log N)\)** time using a divide-and-conquer approach. The most common algorithm is the **Cooley–Tukey radix-2** method, which requires \(N\) to be a power of two. Key steps:

1. **Bit-Reversal Permutation**  
   - Reorder the input array so that indices are arranged in bit-reversed order.  
   - Example for \(N=8\) (binary indices 000, 001, ..., 111):
     - Index 3 (011) ↔ reversed 110 (6)
     - Index 5 (101) ↔ reversed 101 (5), etc.

2. **Butterfly Computations**  
   - For each stage \(s = 1, 2, \ldots, \log_2(N)\):
     - Divide the array into groups of size \(m = 2^s\).
     - Each group has two halves of length \(m/2\).
     - Within each group, perform “butterfly” operations combining pairs of elements separated by \(m/2\) using complex multiplication by twiddle factors \(W_m^k = e^{-\,j \frac{2\pi}{m} k}\).
   - These nested loops over stages and group indices achieve the \(N \log N\) complexity.

In this repository, `fft_cpu.c` implements an in-place Cooley–Tukey FFT in **C** using C99 `<complex.h>`. The GPU version (`fft_gpu.cu`) parallelizes these stages with CUDA kernels.

---

## Repository Structure

```
project/
├── fft.h          # Shared declarations and function prototypes
├── fft_cpu.c      # CPU-only in-place FFT implementation (C)
├── fft_gpu.cu     # GPU kernels and host wrapper for CUDA FFT
├── main.c         # Auto-detect: choose CPU or GPU, run FFT, print results
└── cpu-only/
    ├── fft_cpu.c  # CPU FFT (duplicate for standalone CPU build)
    └── main_cpu.c # CPU-only driver that calls fft_cpu
```

- **fft.h**  
  - Defines `GpuComplex` struct (for GPU code).  
  - Declares `fft_cpu` for CPU FFT.  
  - Optionally declares `gpu_available` and `fft_gpu` when compiling the GPU version.

- **fft_cpu.c**  
  - Contains `void fft_cpu(float _Complex *data, size_t N)`:
    - Computes log2(N), does the bit-reversal permutation, and then the butterfly loops.

- **fft_gpu.cu**  
  - Defines:
    - `__global__ void bit_reversal_permute(GpuComplex *data, size_t N, size_t logN)`.
    - `__global__ void fft_stage_kernel(GpuComplex *data, size_t N, size_t stage)`.
    - `extern "C" int gpu_available(void)` to check for CUDA devices.
    - `extern "C" void fft_gpu(GpuComplex *d_data, size_t N)` to launch kernels and synchronize.

- **main.c**  
  - Generates a test input (sine wave or white noise).  
  - Calls `gpu_available()`:
    - If a CUDA device is found, allocates GPU memory, copies data, times `fft_gpu`, and prints the first 100 bin magnitudes.  
    - Otherwise, calls `fft_cpu` directly on the CPU buffer and prints the first 100 bin magnitudes.

- **cpu-only/**  
  - Provides a standalone CPU build:
    - **fft_cpu.c**: As above, CPU FFT.  
    - **main_cpu.c**: Only calls `fft_cpu` on a generated input (sine wave or noise).

---

## How to Build & Run

### Prerequisites

- **CPU Version** (no GPU required)  
  - A C compiler that supports C99 (e.g., `gcc` or `clang`).  
  - Standard math library (`-lm`).  
  - (Linux/macOS/Windows with WSL).

- **GPU Version** (requires NVIDIA GPU and CUDA)  
  - NVIDIA GPU with compute capability ≥ 3.0.  
  - NVIDIA CUDA Toolkit installed (provides `nvcc`, `cuda_runtime.h`, libraries).  
  - `nvcc` must be in your `PATH`.

---

### 1. CPU-Only Build & Run

```bash
cd project/cpu-only

# Using relative include to parent for fft.h
gcc -O3 -std=c99 -I.. main_cpu.c fft_cpu.c -lm -o fft_cpu

./fft_cpu
```

- **Output**:  
  ```
  CPU FFT completed in XX.XXX ms
  First 100 FFT bins (magnitude):
    Bin   0: ...
    Bin   1: ...
    ...
    Bin  99: ...
  ```

- To use **white noise** instead of a sine wave, edit `main_cpu.c`’s input‐generation loop as demonstrated in the code examples.

---

### 2. Combined CPU/GPU Build & Run

```bash
cd project

# Compile all source files with nvcc (nvcc can link C and CUDA)
nvcc -O3 -std=c99 main.c fft_cpu.c fft_gpu.cu -o fft

./fft
```

- If a CUDA device is present and visible, you’ll see:
  ```
  CUDA device found. Running GPU FFT...
  GPU FFT completed in XX.XXX ms
  First 100 FFT bins (magnitude):
    Bin   0: ...
    Bin   1: ...
    ...
    Bin  99: ...
  ```
- If **no CUDA device** is detected (e.g., on a Mac without NVIDIA GPU), it falls back to:
  ```
  No CUDA device detected. Running CPU FFT...
  CPU FFT completed in YY.YYY ms
  First 100 FFT bins (magnitude):
    Bin   0: ...
    Bin   1: ...
    ...
    Bin  99: ...
  ```

---

## Interpreting the Output

- Each printed “Bin _k_: _magnitude_” corresponds to \(\lvert X[k] \rvert\), the amplitude at frequency index \(k\).  
- A **pure sine** at bin \(k\) (e.g., 50 cycles over \(N\) samples) gives a large magnitude at bin 50 (~\(N/2\)) and near-zero elsewhere.  
- **White noise** in the time domain produces roughly uniform magnitudes across all bins (small random values).

---

## Customizing Input & Analysis

- **Sine Wave Input**  
  - In `main.c` or `main_cpu.c`, change `freq` to any integer from 0 to \(N/2\).  
- **White Noise Input**  
  - Replace the sine‐wave loop with the white‐noise loop:
    ```c
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        float r = (float)rand() / (float)RAND_MAX;  // [0,1]
        r = 2.0f*r - 1.0f;                          // [-1,1]
        cpu_data[i] = r + 0.0f * I;
    }
    ```
- **Normalization**  
  - If you want output magnitudes normalized to 1 for a single sine cycle, divide each `|X[k]|` by `(N/2.0f)` before printing.

---

## Troubleshooting

- **Undefined symbols** when compiling with `gcc`:  
  Make sure you compile both `main_cpu.c` and `fft_cpu.c` together, and include `-I..` if `fft.h` resides in the parent directory.
- **`cudaGetDeviceCount` errors**:  
  Ensure the CUDA Toolkit is installed, `nvcc` is on your `PATH`, and that your GPU drivers are up to date.
- **Wrong magnitudes**:  
  - Confirm `N` is a power of two.  
  - Check that your input is generated as intended (sine or noise).  
  - Make sure you call `fft_cpu` (or `fft_gpu`) with exactly `N` samples.

---

## License

This code is provided under the MIT License. Feel free to use, modify, and distribute.

---

## Acknowledgments

- FFT algorithm based on the Cooley–Tukey radix‑2 approach.  
- CUDA kernels inspired by standard parallel FFT tutorials (bit‐reversal, butterfly steps).  
- CPU code uses C99 `<complex.h>` for clarity and simplicity.

---

Enjoy exploring frequency‐domain analysis on both CPU and GPU!
