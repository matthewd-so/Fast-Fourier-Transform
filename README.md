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

## Performance

NVIDIA T4, CUDA 12.x, N = 2²⁰ complex float samples, average of 50 timed runs (`./bench`):

| Implementation           | Time (ms) | Throughput (GFLOP/s) | Relative          |
| ------------------------ | --------- | -------------------- | ----------------- |
| CPU radix-2 (1 thread)   | ~61       | ~1.7                 | 1×                |
| **This kernel**          | **~0.93** | **~113**             | **~65× vs CPU**   |
| cuFFT                    | ~0.27     | ~385                 | kernel ≈ 29% of cuFFT |

Throughput uses the standard `5·N·log₂(N)` FLOP count for a complex radix-2 FFT. Numbers vary with GPU, clocks, and driver; reproduce on your own hardware with `make bench && ./bench`.

Every number on this page was measured on a free Colab T4, and
[`notebooks/benchmark_colab.ipynb`](notebooks/benchmark_colab.ipynb) re-runs all
of them from a clean clone in about two minutes — no local GPU needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matthewd-so/Fast-Fourier-Transform/blob/main/notebooks/benchmark_colab.ipynb)

The CPU baseline is this repo's own single-threaded radix-2 in `src/fft_cpu.c`,
not FFTW, so read the speedup as "against a straightforward scalar
implementation" rather than against a tuned CPU library. It is also the least
stable number here: on a shared Colab vCPU the same baseline has come out
anywhere between 60 ms and 101 ms, which moves the speedup from ~63× to ~118×.
The table quotes the fastest CPU run, which is the one least flattering to the
GPU.

The ratio against cuFFT is strongly size-dependent, because the T4's L2 is 4 MB
and a complex-float transform is 8·N bytes:

| N | working set | this kernel | cuFFT | ratio |
| --- | --- | --- | --- | --- |
| 2¹⁷ | 1 MB | 0.058 ms | 0.030 ms | 51% |
| 2¹⁸ | 2 MB | 0.072 ms | 0.045 ms | 62% |
| 2¹⁹ | 4 MB | 0.240 ms | 0.102 ms | 43% |
| 2²⁰ | 8 MB | 0.858 ms | 0.212 ms | 25% |
| 2²² | 32 MB | 4.27 ms | 0.966 ms | 23% |

Up to 2 MB the whole array stays resident in L2 and the per-stage passes are
nearly free; past 4 MB every stage becomes a full DRAM round trip, which is
where the pass-count gap against cuFFT starts to bite.

That sweep uses 30 iterations per size. This kernel's own time is steady across
runs, but cuFFT's moves between ~0.21 and ~0.28 ms at N = 2²⁰ depending on where
the clocks settle, so the ratio at that size lands anywhere from 25% to 30%
without anything about either implementation changing.

## How it works

An N-point in-place decimation-in-time transform runs as three kernel types, enqueued back-to-back on one stream with **no host synchronization between stages**:

```
input ──► bit-reversal ──► stages 1..10 fused in shared memory ──► stages 11..log₂N in global memory ──► output
          (1 launch)       (1 launch, one 1024-elem tile/block)    (1 launch per stage)
```

1. **Bit-reversal permutation**: done as a 32×32 tiled transpose through shared memory. Reversing an index is the same operation as swapping its top and bottom 5-bit fields and reversing each, so holding the middle bits fixed leaves a tile whose image is another tile, transposed. Rows are contiguous going in and columns are contiguous coming out, so staging the tile in shared memory makes both ends full 256-byte transactions. Tiles pair off under the permutation, so one block owns both halves of a pair and the swap stays in place with no scratch buffer.
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
`make profile-ncu` (`ncu --section SpeedOfLight`), summed over the twelve
launches:

| Kernel | Launches | GPU time | % of transform | DRAM | GB/s | Compute SOL | Occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fft_bitrev_tiled_kernel` | 1 | 0.078 ms | 8% | 82% | 243 | 16% | 73% |
| `fft_shared_kernel` | 1 | 0.199 ms | 21% | 29% | 92 | 42% | 96% |
| `fft_global_kernel` | 10 | 0.651 ms | 70% | ~92% | ~268 | 14% | ~99% |

Total 0.93 ms, which is what `./bench` reports end to end, so the table accounts
for the whole transform.

Multiplying each kernel's duration by its achieved bandwidth gives the traffic it
actually moved, and that is the number worth reading. A `fft_global_kernel` stage
moves 17.5 MB against a 16.8 MB ideal (8.4 MB in, 8.4 MB out) — essentially
perfect. At ~92% of the T4's peak against 14% compute it is squarely DRAM-bound,
and no arithmetic tuning moves it; cutting the *number* of passes is the only
lever left, which is what the Stockham note under
[Limitations](#limitations) is about.

The whole transform moves ~212 MB per 8 MB array. cuFFT does the same transform
in two kernels and ~42 MB. That 5× traffic ratio, not any per-kernel
inefficiency, is the remaining distance to cuFFT.

### What the two tools turned up

In the order the work happened:

| Profiler evidence | Change | Result |
| --- | --- | --- |
| Timeline: one launch per butterfly stage, the short early stages dominated by launch overhead and repeated DRAM round trips | Fused the first `log₂(1024) = 10` stages into `fft_shared_kernel` — one launch, tile resident in shared memory | 10 launches × ~67 µs → 1 launch × ~78 µs, ~0.59 ms off the transform |
| Timeline: ~15 µs of idle GPU at every stage boundary, waiting on `cudaDeviceSynchronize` | Dropped the per-stage sync; stream ordering already enforces stage order | ~0.3 ms of host-side gap removed across the 21 launches |
| Nsight Compute: separate real/imag loads issuing two 4-byte requests per element | Switched to 8-byte `float2` vectorized loads and stores | 8.0 sectors per request, the ideal for 8-byte accesses; DRAM ~205 → ~237 GB/s |
| Nsight Compute: `sinf`/`cosf` pairs inflating SFU pipe utilization | `__sincosf`, moving twiddle generation onto the SFU with no twiddle table to read | SFU utilization ~31% → ~18% |
| Nsight Compute: achieved occupancy well below theoretical at 128-thread blocks | 512 threads for the tiled kernel (one butterfly per thread per stage), 256 for the elementwise kernels | Achieved occupancy ~55% → ~88% on `fft_global_kernel` |
| Nsight Compute: `fft_bitrev_kernel` was 705 µs of a 1555 µs transform — 45% of the time and the slowest kernel in it, at 174 GB/s against the 268 GB/s the butterfly stages sustain. Duration × bandwidth said it moved ~123 MB to permute an 8 MB buffer: `__brev(i)` lands a stride away, so each 8-byte element dragged in its own 32-byte sector, twice | Rewrote it as a 32×32 tiled transpose staged through shared memory, so both the loads and the stores are full 256-byte transactions | 705 µs → 78 µs, 174 → 243 GB/s, ~123 MB → ~19 MB of traffic. Whole transform 1.45 ms → 0.93 ms, 15% → 29% of cuFFT |

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
