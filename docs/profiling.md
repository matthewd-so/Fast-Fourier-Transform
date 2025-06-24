# Profiling with the Nsight tools

Two tools, two different questions. Nsight Systems answers "where does the wall
clock go across the whole run?" — kernel ordering, gaps, launch overhead, copies,
CPU/GPU overlap. Nsight Compute answers "why is *this* kernel slow?" — hardware
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

- **NVTX ranges** naming each phase — `cpu baseline`, `device setup`,
  `cufft plan`, `warmup`, `timed: this kernel`, `timed: cuFFT`,
  `accuracy checks` — plus per-phase ranges inside the transform itself
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
  the host — a synchronization or launch-bound stage.
- **`cudaDeviceSynchronize` / `cudaLaunchKernel` in the CUDA API row.** Launch
  calls that take longer than the kernels they launch mean the stage is too
  small to be worth its own launch.
- **The stage shape.** `fft_global_kernel` appears once per stage above the
  tile size — 10 launches at N = 2²⁰ — and they should form an even train.
- **`nvtx_pushpop_sum`** in the text summary for the phase-level split, and
  `cuda_gpu_kern_sum` for total GPU time per kernel. The ratio of
  `fft_shared_kernel` to the sum of `fft_global_kernel` launches is the clearest
  signal of where a bigger tile or a Stockham formulation would pay off.

Comparing the `timed: cuFFT` range against `timed: this kernel` on the same
timeline is the honest version of the benchmark table: same GPU, same clocks,
same input.

## Reading the Nsight Compute report

For each `fft_*` kernel, in order:

1. **Speed of Light** — compute vs memory throughput as a percentage of peak.
   The butterfly kernels are memory-bound; if DRAM throughput is high and SM
   throughput is low, arithmetic tweaks are wasted effort.
2. **Memory Workload Analysis** — sectors per request is the coalescing check.
   `float2` accesses with consecutive threads on consecutive elements should
   give 4 sectors per 32-thread request (8 bytes × 32 = 256 B = 4 × 64 B).
   Anything higher means the access pattern regressed.
3. **Occupancy** — achieved vs theoretical. The shared-memory kernel is capped
   by its 8 KB of shared memory per block plus the 512-thread block size; the
   elementwise kernels should sit near the limit at 256 threads.
4. **Warp State Stats** — where warps stall. `Stall Long Scoreboard` dominating
   confirms DRAM latency; `Stall Barrier` points at the `__syncthreads()`
   between shared-memory stages.
5. **Source page** — the build passes `-lineinfo` and the script passes
   `--import-source yes`, so metrics are attributable to individual lines of
   `src/fft_gpu.cu`.

## Changes this workflow drove

The timeline came first in each case; the counters confirmed the fix.

| Observation | Change |
| --- | --- |
| Every stage was a separate launch, and the small early stages were dominated by launch overhead and DRAM round-trips (~2 full passes over the array per stage) | Fused the first `log₂(1024) = 10` stages into `fft_shared_kernel`, one launch, tile resident in shared memory |
| `cudaDeviceSynchronize` between stages showed as host-side gaps on the timeline with the GPU idle | Dropped the per-stage sync; stream ordering already enforces stage order |
| Sectors per request higher than the coalesced ideal with separate real/imag float loads | Switched to 8-byte `float2` vectorized loads and stores |
| `sinf`/`cosf` pairs showing on the SM pipe utilization breakdown | `__sincosf`, moving twiddle generation to the SFU with no twiddle table to read |
| Occupancy below theoretical at the default block sizes | 512 threads for the tiled kernel (one butterfly per thread per stage), 256 for the elementwise kernels, chosen for sm_75 |

## Troubleshooting

- **`ERR_NVGPUCTRPERM`** — Nsight Compute needs permission to read GPU
  performance counters. Run as root, or have an admin allow non-root access
  (`nvidia-smi` docs, NVIDIA knowledge-base article `ERR_NVGPUCTRPERM`).
- **`ncu` takes forever** — kernel replay re-runs each kernel once per pass.
  Lower `NCU_ITERS` (default 2) or keep the `--kernel-name regex:fft_` filter
  rather than reaching for `--set full`.
- **Report names differ between nsys versions** — `nsys stats --help-reports`
  lists what your install supports; older builds use `gpukernsum` / `cudaapisum`
  / `nvtxsum` instead of the `cuda_gpu_kern_sum` style names used in the script.
- **No NVTX rows in the timeline** — the profiled binary is `bench`, not
  `bench_prof`; only the latter is built with `-DFFT_NVTX`.
