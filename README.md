# Fast Fourier Transform in CUDA

A radix-2 Cooley-Tukey FFT written from scratch in CUDA/C, together with a
single-threaded CPU baseline and a benchmark that checks the GPU kernel against
cuFFT for both speed and accuracy.

The project exists to work through what actually makes an FFT fast on a GPU. The
transform itself is textbook decimation-in-time; the interesting part is the
memory hierarchy around it. Early butterfly stages have short spans, so they are
fused into a single kernel that keeps a tile of the signal resident in shared
memory instead of streaming it through DRAM once per stage. Later stages have
spans wide enough that global memory access stays fully coalesced. Everything
runs on one stream with no host synchronization between stages, and both Nsight
Systems and Nsight Compute are wired into the build to check that the design
behaves the way it is supposed to.

It is a learning exercise, not a cuFFT replacement — power-of-two sizes only,
single GPU, complex-to-complex fp32.

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

## Profiling

The benchmark doubles as the profiling harness. `make bench_prof` builds it with
`-DFFT_NVTX`, which turns on NVTX ranges naming each phase of the run and a
`cudaProfilerStart/Stop` bracket around the timed loop; the shipping `./bench`
build compiles all of that away.

```bash
make profile        # Nsight Systems timeline, then Nsight Compute metrics
make profile-nsys   # timeline only
make profile-ncu    # per-kernel counters for the fft_* kernels
make profile-full   # ncu --set full, cuFFT's kernels included
```

Reports and text summaries land in `profiles/`.

- **Nsight Systems** (`nsys`) shows the whole run: the CPU baseline, the H2D
  copy, cuFFT plan creation, and each timed iteration as a labeled NVTX row
  above the CUDA hardware row. It is what makes launch overhead, gaps between
  stages, and the stage-count difference against cuFFT visible.
- **Nsight Compute** (`ncu`) profiles one kernel at a time: speed-of-light,
  sectors per request for the coalescing check, occupancy, and warp stall
  reasons. `-lineinfo` and `--import-source yes` attribute those metrics to
  lines of `src/fft_gpu.cu`.

Changes this drove: fusing the first 10 stages into one shared-memory kernel
(the per-stage version made ~2 full DRAM passes per stage), dropping
`cudaDeviceSynchronize()` between stages once the timeline showed the GPU idling
on host syncs, `float2` vectorized loads and stores, `__sincosf` for twiddles,
and block sizes of 512 (tiled kernel) and 256 (elementwise kernels) for sm_75
occupancy.

[docs/profiling.md](docs/profiling.md) covers the workflow in detail — what to
look at in each report and how to fix the usual permission and version snags.

## Building and running

Requires the CUDA Toolkit (`nvcc`) and an NVIDIA GPU; the benchmark also links against cuFFT (ships with the toolkit). Defaults target sm_75 (T4); override with `make SM=86` etc.

```bash
make            # builds `fft` (demo) and `bench` (benchmark)

./fft           # transforms a 2^20-sample sine wave, verifies the spectral peak
./bench         # this kernel vs cuFFT vs CPU: timings, GFLOP/s, accuracy
./bench 65536 200   # optional: N (power of two) and timed iterations
```

Profiling needs `nsys` and `ncu` on `PATH` (both ship with the toolkit):

```bash
make profile                       # both tools, reports written to profiles/
make profile N=4194304 ITERS=50    # different size / iteration count
```

No NVIDIA GPU (e.g. a Mac)? Build the CPU-only demo:

```bash
make fft_cpu && ./fft_cpu
```

## Accuracy

Everything runs in fp32. `bench` reports the relative L2 error of this kernel against cuFFT and a forward→inverse round-trip error against the original input; both are at the ~1e-6 level for white-noise input at N = 2²⁰. The CPU baseline uses a double-precision twiddle recurrence so it stays a trustworthy reference at large N.

## Repository layout

```
include/fft.h          public API (fft_gpu, ifft_gpu, fft_cpu)
include/fft_profile.h  NVTX / cudaProfilerApi macros, no-ops without -DFFT_NVTX
src/fft_gpu.cu         CUDA kernels + host orchestration
src/fft_cpu.c          single-threaded CPU baseline
src/bench.cu           benchmark vs cuFFT and CPU; doubles as the profiling harness
src/main.c             demo (GPU with CPU fallback; builds CPU-only with -DFFT_CPU_ONLY)
scripts/profile.sh     nsys / ncu runner behind the make profile targets
docs/profiling.md      what to measure and how to read the reports
Makefile               fft / bench / bench_prof / fft_cpu / profile targets
```

## Limitations

- N must be a power of two (radix-2 only, no mixed radix).
- Single-GPU, single-stream, complex-to-complex, fp32 only.
- The global-memory stages are one launch per stage; a Stockham formulation or higher radix would cut passes over DRAM further, and that gap is most of the remaining distance to cuFFT.
