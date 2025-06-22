# Radix-2 Cooley-Tukey FFT in CUDA/C
#
#   make          build the GPU demo (fft)
#   make fft      GPU demo with CPU fallback            (needs nvcc)
#   make fft_cpu  CPU-only demo, no CUDA required       (any C compiler)
#
# SM defaults to 75 (NVIDIA T4). Override for other GPUs, e.g. `make SM=86`.

NVCC ?= nvcc
CC   ?= cc
SM   ?= 75

NVCCFLAGS ?= -O3 -arch=sm_$(SM) -Iinclude --use_fast_math -lineinfo
CFLAGS    ?= -O3 -std=c99 -Iinclude

GPU_SRC = src/fft_gpu.cu src/fft_cpu.c
HDR     = include/fft.h

all: fft

fft: src/main.c $(GPU_SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) src/main.c $(GPU_SRC) -o $@

fft_cpu: src/main.c src/fft_cpu.c $(HDR)
	$(CC) $(CFLAGS) -DFFT_CPU_ONLY src/main.c src/fft_cpu.c -o $@ -lm

clean:
	rm -f fft fft_cpu

.PHONY: all clean
