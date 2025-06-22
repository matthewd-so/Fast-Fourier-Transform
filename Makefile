# Radix-2 Cooley-Tukey FFT in CUDA/C
#
#   make          build the demo (fft) and the benchmark (bench)
#   make fft      GPU demo with CPU fallback            (needs nvcc)
#   make bench    benchmark vs cuFFT and the CPU        (needs nvcc + cuFFT)
#   make fft_cpu  CPU-only demo, no CUDA required       (any C compiler)
#   make profile  Nsight Compute capture of the kernels (needs ncu)
#
# SM defaults to 75 (NVIDIA T4). Override for other GPUs, e.g. `make SM=86`.

NVCC ?= nvcc
CC   ?= cc
SM   ?= 75

NVCCFLAGS ?= -O3 -arch=sm_$(SM) -Iinclude --use_fast_math -lineinfo
CFLAGS    ?= -O3 -std=c99 -Iinclude

GPU_SRC = src/fft_gpu.cu src/fft_cpu.c
HDR     = include/fft.h

all: fft bench

fft: src/main.c $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) src/main.c $(GPU_SRC) -o $@

bench: src/bench.cu $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) src/bench.cu $(GPU_SRC) -o $@ -lcufft

fft_cpu: src/main.c src/fft_cpu.c $(HDR)
	$(CC) $(CFLAGS) -DFFT_CPU_ONLY src/main.c src/fft_cpu.c -o $@ -lm

profile: bench
	ncu --set full -o fft_profile ./bench 1048576 5

clean:
	rm -f fft bench fft_cpu fft_profile.ncu-rep

.PHONY: all profile clean
