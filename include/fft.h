#ifndef FFT_H
#define FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The __host__ __device__ annotations let us use this in both C and CUDA kernels.
typedef struct {
    float real;
    float imag;
} GpuComplex;

int gpu_available(void);

// Perform an in‐place FFT on device memory (d_data points to N GpuComplex values)
void fft_gpu(GpuComplex *d_data, size_t N);

// Perform an in‐place radix‐2 Cooley‐Tukey FFT on the CPU
void fft_cpu(float _Complex *data, size_t N);

// Perform inverse FFT (take frequency domain and return time-domain samples)
void ifft_gpu(GpuComplex *d_data, size_t N);

#ifdef __cplusplus
}
#endif

#endif
