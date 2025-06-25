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

Every optimization in this repo came out of a profiler, in the same two-step
loop each time: Nsight Systems first to find which part of the run was actually
costing wall time, then Nsight Compute on the kernel that turned up, to find out
why. Going straight to the counters tends to mean carefully optimizing a kernel
that was never the problem.

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

- **Nsight Systems** (`nsys`) covers the whole run: the CPU baseline, the H2D
  copy, cuFFT plan creation, and each timed iteration as a labeled NVTX row
  sitting directly above the CUDA hardware row. This is the view that exposed
  the per-stage launch overhead and the host-side gaps where the GPU sat idle,
  and it is where the stage-count difference against cuFFT is obvious.
- **Nsight Compute** (`ncu`) then takes one kernel at a time: speed-of-light,
  sectors per request for the coalescing check, occupancy, and warp stall
  reasons. `-lineinfo` and `--import-source yes` attribute those metrics back to
  individual lines of `src/fft_gpu.cu`, which is what made the memory-access
  fixes below straightforward to find.

### Where the time goes

Per-kernel breakdown of one N = 2²⁰ transform on the T4, from
`nsys stats --report cuda_gpu_kern_sum` with the speed-of-light figures from the
matching Nsight Compute report:

| Kernel | Launches | GPU time | % of transform | Memory SOL | Compute SOL |
| --- | --- | --- | --- | --- | --- |
| `fft_bitrev_kernel` | 1 | ~0.10 ms | ~11% | 38% | 6% |
| `fft_shared_kernel` | 1 | ~0.08 ms | ~8% | 62% | 24% |
| `fft_global_kernel` | 10 | ~0.67 ms | ~71% | 74% | 11% |

The remaining ~10% is launch and enqueue gap between the twelve kernels. The
shape of that table is the whole story of the design: one fused shared-memory
launch retires ten butterfly stages in 0.08 ms, while the ten global-memory
stages that follow cost 0.67 ms between them. At 74% of the T4's 320 GB/s peak
against 11% compute, `fft_global_kernel` is squarely DRAM-bound — every stage is
a full 16 MB round trip through memory, and no amount of arithmetic tuning moves
it. Cutting the *number* of passes is the only lever left, which is what the
Stockham note under [Limitations](#limitations) is about.

### What the two tools turned up

In the order the work happened:

| Profiler evidence | Change | Result |
| --- | --- | --- |
| Timeline: one launch per butterfly stage, the short early stages dominated by launch overhead and repeated DRAM round trips | Fused the first `log₂(1024) = 10` stages into `fft_shared_kernel` — one launch, tile resident in shared memory | 10 launches × ~67 µs → 1 launch × ~78 µs, ~0.59 ms off the transform |
| Timeline: ~15 µs of idle GPU at every stage boundary, waiting on `cudaDeviceSynchronize` | Dropped the per-stage sync; stream ordering already enforces stage order | ~0.3 ms of host-side gap removed across the 21 launches |
| Nsight Compute: separate real/imag loads issuing two 4-byte requests per element | Switched to 8-byte `float2` vectorized loads and stores | 8.0 sectors per request, the ideal for 8-byte accesses; DRAM ~205 → ~237 GB/s |
| Nsight Compute: `sinf`/`cosf` pairs inflating SFU pipe utilization | `__sincosf`, moving twiddle generation onto the SFU with no twiddle table to read | SFU utilization ~31% → ~18% |
| Nsight Compute: achieved occupancy well below theoretical at 128-thread blocks | 512 threads for the tiled kernel (one butterfly per thread per stage), 256 for the elementwise kernels | Achieved occupancy ~55% → ~88% on `fft_global_kernel` |

Warp stall reasons confirm the split: `fft_global_kernel` sits on
`Stall Long Scoreboard` (DRAM latency), while `fft_shared_kernel`'s dominant
stall is `Stall Barrier` — the `__syncthreads()` between its ten stages, which
is the price of keeping the tile in shared memory and a much cheaper price than
the round trip it replaced.

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
