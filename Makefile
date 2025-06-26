# Radix-2 Cooley-Tukey FFT in CUDA/C
#
#   make               build the demo (fft) and the benchmark (bench)
#   make fft           GPU demo with CPU fallback              (needs nvcc)
#   make bench         benchmark vs cuFFT and the CPU          (needs nvcc + cuFFT)
#   make fft_cpu       CPU-only demo, no CUDA required         (any C compiler)
#
#   make bench_prof    benchmark with NVTX ranges compiled in  (needs nvcc)
#   make profile       Nsight Systems timeline, then Nsight Compute metrics
#   make profile-nsys  Nsight Systems only                     (needs nsys)
#   make profile-ncu   Nsight Compute, fft_* kernels only      (needs ncu)
#   make profile-full  Nsight Compute --set full, all kernels including cuFFT
#
# SM defaults to 75 (NVIDIA T4). Override for other GPUs, e.g. `make SM=86`.
# Profiling knobs: `make profile N=4194304 ITERS=20 PROFDIR=profiles`.
# NCU_ITERS is the timed-iteration count for the Nsight Compute passes only,
# kept separate because kernel replay makes each iteration expensive.
# See docs/profiling.md for what to read in each report.

NVCC ?= nvcc
CC   ?= cc
SM   ?= 75

NVCCFLAGS ?= -O3 -arch=sm_$(SM) -Iinclude --use_fast_math -lineinfo
CFLAGS    ?= -O3 -std=c99 -Iinclude

# Profiling defaults, passed through to scripts/profile.sh.
N         ?= 1048576
ITERS     ?= 20
NCU_ITERS ?= 2
PROFDIR   ?= profiles
PROFILE    = BENCH=./bench_prof N=$(N) ITERS=$(ITERS) NCU_ITERS=$(NCU_ITERS) \
             PROFDIR=$(PROFDIR) ./scripts/profile.sh

GPU_SRC = src/fft_gpu.cu src/fft_cpu.c
HDR     = include/fft.h include/fft_profile.h

all: fft bench

fft: src/main.c $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) src/main.c $(GPU_SRC) -o $@

bench: src/bench.cu $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) src/bench.cu $(GPU_SRC) -o $@ -lcufft

# Same benchmark, plus NVTX ranges and the cudaProfilerStart/Stop bracket that
# lets Nsight Compute skip warmup. Kept as a separate binary so the timings
# reported by ./bench are never taken from an instrumented build.
bench_prof: src/bench.cu $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) -DFFT_NVTX src/bench.cu $(GPU_SRC) -o $@ -lcufft

fft_cpu: src/main.c src/fft_cpu.c $(HDR)
	$(CC) $(CFLAGS) -DFFT_CPU_ONLY src/main.c src/fft_cpu.c -o $@ -lm

profile: bench_prof
	$(PROFILE) all

profile-nsys: bench_prof
	$(PROFILE) nsys

profile-ncu: bench_prof
	$(PROFILE) ncu

profile-full: bench_prof
	$(PROFILE) ncu-full

clean:
	rm -f fft bench bench_prof fft_cpu fft_profile.ncu-rep
	rm -rf $(PROFDIR)

.PHONY: all profile profile-nsys profile-ncu profile-full clean
