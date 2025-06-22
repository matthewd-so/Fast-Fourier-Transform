# Fast Fourier Transform in CUDA

A radix-2 Cooley-Tukey FFT kernel implemented in CUDA/C, using **shared-memory tiling** and **coalesced global-memory access** to reach **~25% of cuFFT throughput** and a **~30× speedup over the CPU baseline** on 1M-point transforms (NVIDIA T4), tuned with **Nsight Compute**.

## Performance

NVIDIA T4, CUDA 12.x, N = 2²⁰ complex float samples, average of 50 timed runs (`./bench`):

| Implementation           | Time (ms) | Throughput (GFLOP/s) | Relative          |
| ------------------------ | --------- | -------------------- | ----------------- |
| CPU radix-2 (1 thread)   | ~28.5     | ~3.7                 | 1×                |
| **This kernel**          | **~0.95** | **~110**             | **~30× vs CPU**   |
| cuFFT                    | ~0.24     | ~437                 | kernel ≈ 25% of cuFFT |

Throughput uses the standard `5·N·log₂(N)` FLOP count for a complex radix-2 FFT. Numbers vary with GPU, clocks, and driver; reproduce on your own hardware with `make bench && ./bench`.

## How it works

An N-point in-place decimation-in-time transform runs as three kernel types, enqueued back-to-back on one stream with **no host synchronization between stages**:

```
input ──► bit-reversal ──► stages 1..10 fused in shared memory ──► stages 11..log₂N in global memory ──► output
          (1 launch)       (1 launch, one 1024-elem tile/block)    (1 launch per stage)
```

1. **Bit-reversal permutation**: each thread computes its partner index with the hardware `__brev` intrinsic and swaps once.
2. **Shared-memory tiled stages**: after bit reversal, the first `log₂(1024) = 10` butterfly stages only combine elements within a contiguous 1024-element tile. Each block loads its tile into shared memory with coalesced `float2` reads, runs all 10 stages entirely in shared memory (8 KB per block, `__syncthreads()` between stages), and writes back coalesced. Those stages never touch DRAM between butterflies, and 10 kernel launches collapse into 1.
3. **Global-memory stages**: the remaining stages have butterfly spans ≥ 1024 elements, so consecutive threads read/write consecutive 8-byte `float2` elements in both halves of each group: every global access is fully coalesced.

Twiddle factors are computed on the fly by the SFU via `__sincosf`, trading a few ULP of accuracy for zero twiddle-table bandwidth.

### Tuning notes (Nsight Compute)

The Makefile builds with `-lineinfo` so `ncu` correlates metrics to source (`make profile`). Changes driven by profiling:

- Fusing the first 10 stages into the shared-memory kernel: the per-stage version was DRAM-bound with ~2 full passes over the array per stage.
- Removing `cudaDeviceSynchronize()` between stages: stream ordering already guarantees correctness, and launch/sync overhead dominated small-stage kernels.
- `float2` (8-byte) vectorized loads/stores instead of separate real/imag floats.
- `__sincosf` instead of `sinf`/`cosf` pairs, moving twiddle generation to the SFU.
- Block sizes: 512 threads for the tiled kernel (one butterfly per thread per stage), 256 for the elementwise kernels, chosen for occupancy on sm_75.

## Building and running

Requires the CUDA Toolkit (`nvcc`) and an NVIDIA GPU; the benchmark also links against cuFFT (ships with the toolkit). Defaults target sm_75 (T4); override with `make SM=86` etc.

```bash
make            # builds `fft` (demo) and `bench` (benchmark)

./fft           # transforms a 2^20-sample sine wave, verifies the spectral peak
./bench         # this kernel vs cuFFT vs CPU: timings, GFLOP/s, accuracy
./bench 65536 200   # optional: N (power of two) and timed iterations

make profile    # Nsight Compute capture of all kernels (writes fft_profile.ncu-rep)
```

No NVIDIA GPU (e.g. a Mac)? Build the CPU-only demo:

```bash
make fft_cpu && ./fft_cpu
```

## Accuracy

Everything runs in fp32. `bench` reports the relative L2 error of this kernel against cuFFT and a forward→inverse round-trip error against the original input; both are at the ~1e-6 level for white-noise input at N = 2²⁰. The CPU baseline uses a double-precision twiddle recurrence so it stays a trustworthy reference at large N.

## Repository layout

```
include/fft.h     public API (fft_gpu, ifft_gpu, fft_cpu)
src/fft_gpu.cu    CUDA kernels + host orchestration
src/fft_cpu.c     single-threaded CPU baseline
src/bench.cu      benchmark vs cuFFT and CPU
src/main.c        demo (GPU with CPU fallback; builds CPU-only with -DFFT_CPU_ONLY)
Makefile          fft / bench / fft_cpu / profile targets
```

## Limitations

- N must be a power of two (radix-2 only, no mixed radix).
- Single-GPU, single-stream, complex-to-complex, fp32 only.
- The global-memory stages are one launch per stage; a Stockham formulation or higher radix would cut passes over DRAM further, and that gap is most of the remaining distance to cuFFT.
