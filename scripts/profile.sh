#!/usr/bin/env bash
#
# Drive the Nsight tools over the FFT benchmark.
#
#   scripts/profile.sh nsys       Nsight Systems: whole-run timeline + summaries
#   scripts/profile.sh ncu        Nsight Compute: metrics for the fft_* kernels
#   scripts/profile.sh ncu-full   Nsight Compute: every section, every kernel
#   scripts/profile.sh all        nsys then ncu (timeline first, then counters)
#
# Environment overrides:
#   BENCH=./bench_prof   instrumented benchmark binary (make bench_prof)
#   N=1048576            transform size
#   ITERS=20             timed iterations for the nsys run
#   NCU_ITERS=2          timed iterations for the ncu runs (kernel replay is slow)
#   PROFDIR=profiles     where reports and text summaries are written
#
# Reports land in $PROFDIR: open the .nsys-rep / .ncu-rep in the Nsight GUIs,
# or read the .txt summaries this script writes alongside them.

set -euo pipefail

BENCH=${BENCH:-./bench_prof}
N=${N:-1048576}
ITERS=${ITERS:-20}
NCU_ITERS=${NCU_ITERS:-2}
PROFDIR=${PROFDIR:-profiles}
NSYS=${NSYS:-nsys}
NCU=${NCU:-ncu}

die() { printf 'profile.sh: %s\n' "$*" >&2; exit 1; }

require_tool() {
    command -v "$1" >/dev/null 2>&1 || die \
        "$1 not found in PATH; it ships with the CUDA Toolkit, or standalone from developer.nvidia.com/nsight-$2"
}

require_bench() {
    [ -x "$BENCH" ] || die "$BENCH not found. Build it with: make bench_prof"
    mkdir -p "$PROFDIR"
}

# Nsight Systems: full run, no capture range. The point of the timeline is to
# see everything -- CPU baseline, H2D copy, plan creation, warmup, timed loops --
# and where the gaps between them are.
run_nsys() {
    require_tool "$NSYS" systems
    require_bench
    echo "==> Nsight Systems: $BENCH $N $ITERS"
    "$NSYS" profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output "$PROFDIR/fft_timeline" \
        "$BENCH" "$N" "$ITERS"

    # Kernel time by name, CUDA API time by call, and wall time per NVTX phase.
    "$NSYS" stats --force-export=true \
        --report cuda_gpu_kern_sum \
        --report cuda_gpu_mem_time_sum \
        --report cuda_api_sum \
        --report nvtx_pushpop_sum \
        "$PROFDIR/fft_timeline.nsys-rep" | tee "$PROFDIR/fft_timeline.txt"
    echo "==> wrote $PROFDIR/fft_timeline.nsys-rep and .txt"
}

# Nsight Compute: per-kernel hardware counters. --profile-from-start off pairs
# with the cudaProfilerStart/Stop bracket in bench.cu, so warmup and the
# correctness checks are not replayed.
run_ncu() {
    require_tool "$NCU" compute
    require_bench
    echo "==> Nsight Compute (fft_* kernels): $BENCH $N $NCU_ITERS"
    "$NCU" \
        --profile-from-start off \
        --kernel-name regex:fft_ \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section LaunchStats \
        --section SchedulerStats \
        --section WarpStateStats \
        --import-source yes \
        --force-overwrite \
        --export "$PROFDIR/fft_kernels" \
        "$BENCH" "$N" "$NCU_ITERS" | tee "$PROFDIR/fft_kernels.txt"
    echo "==> wrote $PROFDIR/fft_kernels.ncu-rep and .txt"
}

# The everything pass: all sections, cuFFT's kernels included, for side-by-side
# comparison. Minutes, not seconds.
run_ncu_full() {
    require_tool "$NCU" compute
    require_bench
    echo "==> Nsight Compute (--set full, all kernels): $BENCH $N 1"
    "$NCU" \
        --profile-from-start off \
        --set full \
        --import-source yes \
        --force-overwrite \
        --export "$PROFDIR/fft_full" \
        "$BENCH" "$N" 1 | tee "$PROFDIR/fft_full.txt"
    echo "==> wrote $PROFDIR/fft_full.ncu-rep and .txt"
}

case "${1:-all}" in
    nsys)     run_nsys ;;
    ncu)      run_ncu ;;
    ncu-full) run_ncu_full ;;
    all)      run_nsys; run_ncu ;;
    *)        die "unknown mode '$1' (expected: nsys | ncu | ncu-full | all)" ;;
esac
