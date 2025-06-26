# Profiling with the Nsight tools

Two tools, two different questions. Nsight Systems answers "where does the wall
clock go across the whole run?": kernel ordering, gaps, launch overhead, copies,
CPU/GPU overlap. Nsight Compute answers "why is *this* kernel slow?": hardware
counters for one kernel at a time, correlated back to source lines.

Run them in that order. A timeline first tells you which kernel is worth opening
in Nsight Compute; going straight to counters usually means optimizing a kernel
that wasn't the problem.

## Setup

```bash
make bench_prof     # benchmark compiled with -DFFT_NVTX
make profile        # nsys timeline, then ncu on the fft_* kernels
```

`make profile` shells out to `scripts/profile.sh`, which writes reports and
plain-text summaries into `profiles/`. Individually:

```bash
make profile-nsys                  # timeline only
make profile-ncu                   # counters for the fft_* kernels only
make profile-full                  # ncu --set full, cuFFT's kernels included
make profile N=4194304 ITERS=50    # different size / iteration count
```

Open `profiles/fft_timeline.nsys-rep` in Nsight Systems and
`profiles/fft_kernels.ncu-rep` in Nsight Compute, or read the `.txt` files next
to them.

### What the instrumentation does

`include/fft_profile.h` compiles to nothing unless `FFT_NVTX` is defined, so
`./bench` and `./fft` carry no profiling overhead. In `bench_prof` it enables:

- **NVTX ranges** naming each phase (`cpu baseline`, `device setup`,
  `cufft plan`, `warmup`, `timed: this kernel`, `timed: cuFFT`,
  `accuracy checks`), plus per-phase ranges inside the transform itself
  (`bitrev`, `shared stages`, `global stages`). The ranges inside `fft_gpu` are
  host-side, so they measure *enqueue* cost, not kernel duration; their value is
  that they sit directly above the CUDA HW row in the timeline, which is what
  makes launch overhead and inter-stage gaps visible.
- **`cudaProfilerStart/Stop`** around the timed loop only. Nsight Compute runs
  with `--profile-from-start off`, so it replays steady-state kernels and skips
  the warmup runs and correctness checks. Without this, `--set full` over 50
  iterations of a dozen launches takes minutes and reports cold-cache numbers.

The benchmark restores the working buffer from a pristine device copy before
every run, so profiled iterations are all doing identical work.

## Reading the Nsight Systems timeline

Zoom into one iteration of the `timed: this kernel` range and look at:

- **Gaps between kernels on the CUDA HW row.** Back-to-back launches on one
  stream should be nearly contiguous. Visible gaps mean the GPU is waiting on
  the host: a synchronization or launch-bound stage.
- **`cudaDeviceSynchronize` / `cudaLaunchKernel` in the CUDA API row.** Launch
  calls that take longer than the kernels they launch mean the stage is too
  small to be worth its own launch.
- **The stage shape.** `fft_global_kernel` appears once per stage above the
  tile size (10 launches at N = 2²⁰) and they should form an even train.
- **`nvtx_pushpop_sum`** in the text summary for the phase-level split, and
  `cuda_gpu_kern_sum` for total GPU time per kernel. The ratio of
  `fft_shared_kernel` to the sum of `fft_global_kernel` launches is the clearest
  signal of where a bigger tile or a Stockham formulation would pay off.

Comparing the `timed: cuFFT` range against `timed: this kernel` on the same
timeline is the honest version of the benchmark table: same GPU, same clocks,
same input.

## Reading the Nsight Compute report

For each `fft_*` kernel, in order:

1. **Speed of Light**: compute vs memory throughput as a percentage of peak.
   The butterfly kernels are memory-bound; if DRAM throughput is high and SM
   throughput is low, arithmetic tweaks are wasted effort.
2. **Memory Workload Analysis**: sectors per request is the coalescing check.
   `float2` accesses with consecutive threads on consecutive elements give
   8 sectors per 32-thread request (8 B × 32 = 256 B, over 32 B sectors), and
   that is what the butterfly kernels measure. Anything higher means the access
   pattern regressed. `fft_bitrev_tiled_kernel` is held to the same standard:
   the tiled transpose exists precisely so that the reversed index does not
   scatter. `fft_bitrev_kernel`, the fallback for n < 1024, does scatter by
   construction, but only ever runs on an array that fits in L2.
3. **Occupancy**: achieved vs theoretical. The shared-memory kernel is capped
   by its 8 KB of shared memory per block plus the 512-thread block size; the
   elementwise kernels should sit near the limit at 256 threads.
4. **Warp State Stats**: where warps stall. `Stall Long Scoreboard` dominating
   confirms DRAM latency; `Stall Barrier` points at the `__syncthreads()`
   between shared-memory stages.
5. **Source page**: the build passes `-lineinfo` and the script passes
   `--import-source yes`, so metrics are attributable to individual lines of
   `src/fft_gpu.cu`.

## Baseline numbers

For comparison when you re-profile, one N = 2²⁰ transform on a T4
(`make profile-ncu`, driver 580.82.07, CUDA 12.x):

| Kernel | Launches | GPU time | % of transform | DRAM | GB/s | Compute SOL | Occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fft_bitrev_tiled_kernel` | 1 | 0.078 ms | 8% | 82% | 243 | 16% | 73% |
| `fft_shared_kernel` | 1 | 0.199 ms | 21% | 29% | 92 | 42% | 96% |
| `fft_global_kernel` | 10 | 0.651 ms | 70% | ~92% | ~268 | 14% | ~99% |

The twelve launches sum to 0.93 ms, which is what `./bench` reports end to end.

`fft_global_kernel` at ~92% DRAM against 14% compute is the headline: the
transform is DRAM-bound in its back half, so the remaining wins are in reducing
the number of passes over the array, not in the arithmetic. The useful cross
check is duration × achieved bandwidth, which gives the traffic a kernel really
moved. 17.5 MB per global stage against a 16.8 MB ideal is as good as that pass
gets, and it was a 123 MB reading on the old scatter-based bit reversal that
justified rewriting it as a transpose.

## Measuring what an optimization is worth

`make ablate` rebuilds `src/fft_gpu.cu` once per `FFT_ABLATE_*` switch, each of
which restores one pre-optimization behaviour, and runs the benchmark against
every variant. The accuracy checks run in every build, so a variant that breaks
correctness shows up as a bad L2 error instead of a fast time.

```bash
make ablate                       # N = 2^20, 50 iterations
make ablate N=4194304 ITERS=20    # larger transform
```

Baseline 0.942 / 0.963 ms at N = 2²⁰ on a T4 over two independent runs; "vs
base" is transform time relative to the shipping build, so higher is slower.
Run it twice before believing any row inside a few percent.

| Switch | Puts back | vs base |
| --- | --- | --- |
| `FFT_ABLATE_SCATTER` | element-at-a-time bit reversal | 1.65–1.69× |
| `FFT_ABLATE_NO_FUSION` | one launch per butterfly stage | 1.51–1.54× |
| `FFT_ABLATE_SYNC` | `cudaDeviceSynchronize` between stages | 1.03–1.11× |
| `FFT_ABLATE_BLOCK128` | 128-thread blocks | 1.00–1.02× |
| `FFT_ABLATE_SINCOS` | `sinf`/`cosf` instead of `__sincosf` | 0.96–1.02× |
| `FFT_ABLATE_SCALAR` | 4-byte real/imag loads | 0.97–0.98× |

The two memory-traffic changes carry the design. The bottom three sit inside
run-to-run noise, and the counters say why, measured on `fft_global_kernel`
with `--launch-count 1`:

| Variant | DRAM GB/s | sectors/request | XU pipe | occupancy |
| --- | --- | --- | --- | --- |
| baseline | 269.35 | 8 | 13.61% | 90.28% |
| `SCALAR` | 269.51 | 8 | 13.61% | 89.48% |
| `SINCOS` | 269.43 | 8 | 13.70% | 90.27% |
| `BLOCK128` | 269.04 | 8 | 13.54% | 90.56% |

Nothing moves. nvcc emits the same 8-sector accesses with or without the
hand-written `float2`; `--use_fast_math` already lowers `sinf`/`cosf` to the
same SFU instructions `__sincosf` uses; and occupancy on this kernel is set by
the grid, not the block size. Worth keeping in mind before crediting a
micro-optimization that was never measured.

## Troubleshooting

- **`ERR_NVGPUCTRPERM`**: Nsight Compute needs permission to read GPU
  performance counters. Run as root, or have an admin allow non-root access
  (`nvidia-smi` docs, NVIDIA knowledge-base article `ERR_NVGPUCTRPERM`).
- **`ncu` takes forever**: kernel replay re-runs each kernel once per pass.
  Lower `NCU_ITERS` (default 2) or keep the `--kernel-name regex:fft_` filter
  rather than reaching for `--set full`.
- **Report names differ between nsys versions**: `nsys stats --help-reports`
  lists what your install supports; older builds use `gpukernsum` / `cudaapisum`
  / `nvtxsum` instead of the `cuda_gpu_kern_sum` style names used in the script.
- **No NVTX rows in the timeline**: the profiled binary is `bench`, not
  `bench_prof`; only the latter is built with `-DFFT_NVTX`.
