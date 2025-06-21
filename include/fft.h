#ifndef FFT_H
#define FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved single-precision complex sample.
 * Layout-compatible with CUDA's float2 and cuFFT's cufftComplex. */
typedef struct {
    float real;
    float imag;
} GpuComplex;

/* Returns 1 if a CUDA-capable device is present, 0 otherwise. */
int gpu_available(void);

/* In-place radix-2 Cooley-Tukey FFT on device memory.
 * d_data points to N complex samples on the GPU; N must be a power of two.
 * Kernels are enqueued on the default stream; synchronize to get results. */
void fft_gpu(GpuComplex *d_data, size_t N);

/* In-place inverse FFT on device memory. Output is scaled by 1/N. */
void ifft_gpu(GpuComplex *d_data, size_t N);

/* In-place radix-2 Cooley-Tukey FFT on the CPU (single-threaded baseline). */
void fft_cpu(float _Complex *data, size_t N);

#ifdef __cplusplus
}
#endif

#endif
