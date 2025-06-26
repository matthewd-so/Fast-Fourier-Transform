#!/usr/bin/env bash
#
# Measure what each optimization is worth, by putting the pre-optimization
# behaviour back one at a time and re-running the benchmark.
#
#   scripts/ablate.sh            N=1048576 ITERS=50 by default
#   N=4194304 scripts/ablate.sh  larger transform
#
# Each variant is a full rebuild of src/fft_gpu.cu with one FFT_ABLATE_* macro
# defined; see the switch list at the top of that file. The baseline is the
# shipping build with none of them set. Every variant still runs the benchmark's
# accuracy checks, so a change that breaks correctness shows up as a bad L2
# error rather than a misleadingly fast time.

set -euo pipefail

NVCC=${NVCC:-nvcc}
SM=${SM:-75}
N=${N:-1048576}
ITERS=${ITERS:-50}
OUTDIR=${OUTDIR:-ablate}

FLAGS="-O3 -arch=sm_$SM -Iinclude --use_fast_math -lineinfo"
SRC="src/bench.cu src/fft_gpu.cu src/fft_cpu.c"

# name|macro|what it puts back
VARIANTS=(
    "baseline|                        |shipping build, all optimizations on"
    "no-fusion|-DFFT_ABLATE_NO_FUSION |one launch per butterfly stage"
    "sync|-DFFT_ABLATE_SYNC           |cudaDeviceSynchronize between stages"
    "scalar|-DFFT_ABLATE_SCALAR       |4-byte real/imag loads, not float2"
    "sincos|-DFFT_ABLATE_SINCOS       |sinf/cosf instead of __sincosf"
    "block128|-DFFT_ABLATE_BLOCK128   |128-thread blocks"
    "scatter|-DFFT_ABLATE_SCATTER     |element-at-a-time bit reversal"
)

mkdir -p "$OUTDIR"
printf 'Ablation: N=%s, %s iterations, sm_%s\n\n' "$N" "$ITERS" "$SM"
printf '%-10s  %9s  %10s  %8s  %-12s  %s\n' \
    variant time GFLOP/s "vs base" "L2 error" "puts back"
printf '%s\n' "----------------------------------------------------------------------------------------"

base_ms=""
for v in "${VARIANTS[@]}"; do
    name=${v%%|*}; rest=${v#*|}
    macro=$(echo "${rest%%|*}" | xargs || true)
    desc=${rest#*|}

    $NVCC $FLAGS $macro $SRC -o "$OUTDIR/bench_$name" -lcufft 2>/dev/null

    out=$("$OUTDIR/bench_$name" "$N" "$ITERS")
    ms=$(echo "$out" | awk '/This kernel/ {print $3}')
    gf=$(echo "$out" | awk '/This kernel/ {print $5}')
    l2=$(echo "$out" | awk '/Relative L2 error/ {print $NF}')

    if [ -z "$base_ms" ]; then base_ms=$ms; rel="--"; else
        rel=$(awk -v a="$ms" -v b="$base_ms" 'BEGIN{printf "%.2fx", a/b}')
    fi
    printf '%-10s  %7s ms  %10s  %8s  %-12s  %s\n' "$name" "$ms" "$gf" "$rel" "$l2" "$desc"
done

printf '\n"vs base" is transform time relative to the shipping build; higher is slower.\n'
printf 'Binaries left in %s/ if you want to run ncu against a variant.\n' "$OUTDIR"
