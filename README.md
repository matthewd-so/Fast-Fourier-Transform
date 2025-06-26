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
| CPU radix-2 (1 thread)   | ~60       | ~1.7                 | 1×                |
| **This kernel**          | **~0.86** | **~122**             | **~70× vs CPU**   |
| cuFFT                    | ~0.21     | ~499                 | kernel ≈ 25% of cuFFT |

Throughput uses the standard `5·N·log₂(N)` FLOP count for a complex radix-2 FFT. Numbers vary with GPU, clocks, and driver; reproduce on your own hardware with `make bench && ./bench`.

Every number on this page was measured on a free Colab T4, and
[`notebooks/benchmark_colab.ipynb`](notebooks/benchmark_colab.ipynb) re-runs all
of them from a clean clone in about two minutes, with no local GPU needed:

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

`make bench_prof` builds the benchmark with `-DFFT_NVTX`, adding NVTX ranges per
phase and a `cudaProfilerStart/Stop` bracket around the timed loop. The shipping
`./bench` build compiles all of it away.

```bash
make profile        # Nsight Systems timeline, then Nsight Compute metrics
make profile-nsys   # timeline only
make profile-ncu    # per-kernel counters for the fft_* kernels
make profile-full   # ncu --set full, cuFFT's kernels included
make ablate         # rebuild with each optimization removed, tabulate the cost
```

Reports land in `profiles/`. [docs/profiling.md](docs/profiling.md) covers what
to read in each one.

### Where the time goes

Per-kernel breakdown of one N = 2²⁰ transform on the T4, from
`make profile-ncu` (`ncu --section SpeedOfLight`), summed over the twelve
launches:

| Kernel | Launches | GPU time | % of transform | DRAM | GB/s | Compute SOL | Occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fft_bitrev_tiled_kernel` | 1 | 0.078 ms | 8% | 82% | 243 | 16% | 73% |
| `fft_shared_kernel` | 1 | 0.199 ms | 21% | 29% | 92 | 42% | 96% |
| `fft_global_kernel` | 10 | 0.651 ms | 70% | ~92% | ~268 | 14% | ~99% |

The twelve launches sum to 0.93 ms, which is what `./bench` reports end to end.

Duration × achieved bandwidth gives the traffic a kernel actually moved. A
`fft_global_kernel` stage moves 17.5 MB against a 16.8 MB ideal (8.4 in, 8.4
out), so the access pattern is as good as that pass gets. At ~92% of peak
against 14% compute it is DRAM-bound, and only cutting the number of passes
moves it; see [Limitations](#limitations).

The whole transform moves ~212 MB per 8 MB array; cuFFT does it in two kernels
and ~42 MB. That 5× traffic ratio is the remaining distance.

### What each optimization is actually worth

`make ablate` rebuilds the kernel with one optimization removed at a time and
re-runs the benchmark, so the cost of each is measured rather than asserted.
N = 2²⁰, 50 iterations, T4:

Two independent runs, since anything inside a few percent is run-to-run noise:

| Optimization removed | Run 1 | Run 2 | vs baseline |
| --- | --- | --- | --- |
| *(none, shipping build)* | 0.942 ms | 0.963 ms | baseline |
| Tiled bit-reversal → element-at-a-time scatter | 1.588 ms | 1.589 ms | **1.65–1.69×** |
| Shared-memory fusion → one launch per stage | 1.447 ms | 1.450 ms | **1.51–1.54×** |
| Stream ordering → `cudaDeviceSynchronize` per stage | 1.046 ms | 0.993 ms | 1.03–1.11× |
| 256-thread blocks → 128 | 0.961 ms | 0.967 ms | 1.00–1.02× |
| `__sincosf` → `sinf`/`cosf` | 0.960 ms | 0.926 ms | 0.96–1.02× |
| `float2` loads → separate 4-byte real/imag | 0.913 ms | 0.943 ms | 0.97–0.98× |

Two changes carry the design, and both reproduce tightly: coalescing the bit
reversal is worth ~1.67× and fusing the first ten stages into shared memory
~1.52×. Both attack the same thing: how many times the array crosses the
memory bus. Removing the per-stage sync is worth something, but it straddles
the noise floor at 1.03–1.11×, well short of the ~0.3 ms once claimed for it.

The bottom three land within noise of the baseline, one of them on both sides
of it, and the counters say why:

- **`float2` vectorization**: no effect. Both builds report 8.0 sectors per
  request and 269 GB/s on `fft_global_kernel`. nvcc emits the same access
  pattern either way, so the hand-vectorization buys nothing here.
- **`__sincosf`**: no effect, because `--use_fast_math` already maps `sinf` and
  `cosf` onto the same SFU instructions. XU pipe utilization is 13.6% against
  13.7%. The intrinsic is redundant with the build flag, not with the hardware.
- **Block size**: no effect. `fft_global_kernel` occupancy is 90.3% at 256
  threads and 90.6% at 128; it is limited by the grid, not the block.

Warp stalls confirm the split: `fft_global_kernel` sits on
`Stall Long Scoreboard` (DRAM latency), `fft_shared_kernel` on `Stall Barrier`
(the `__syncthreads()` between its ten stages).

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
make ablate                        # cost of each optimization (nvcc only)
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
scripts/ablate.sh      rebuilds with each optimization removed, behind make ablate
notebooks/             Colab notebook that reproduces the benchmark table
docs/profiling.md      what to measure and how to read the reports
Makefile               fft / bench / bench_prof / fft_cpu / profile / ablate targets
```

## Limitations

- N must be a power of two (radix-2 only, no mixed radix).
- Single-GPU, single-stream, complex-to-complex, fp32 only.
- The global-memory stages are one launch per stage; a Stockham formulation or higher radix would cut passes over DRAM further, and that gap is most of the remaining distance to cuFFT.
