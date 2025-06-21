#include "fft.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// We can reuse the bit_reversal_permute kernel from fft_gpu.cu
extern __global__ void bit_reversal_permute(GpuComplex *data, size_t N, size_t logN);
extern __global__ void fft_stage_kernel(GpuComplex *data, size_t N, size_t stage);

// Inverse FFT stage kernel: same as forward but with +2π instead of -2π
__global__
void ifft_stage_kernel(GpuComplex *data, size_t N, size_t stage) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = 1U << stage;
    size_t half = m >> 1;
    size_t total_pairs = N >> 1;
    if (idx >= total_pairs) return;

    size_t group = idx / half;
    size_t pos   = idx % half;
    size_t i = group * m + pos;
    size_t j = i + half;

    float angle = +2.0f * M_PI * ((float)pos) / ((float)m);
    GpuComplex w = {cosf(angle), sinf(angle)};

    GpuComplex u = data[i];
    GpuComplex v = data[j];
    // t = w * v
    GpuComplex t;
    t.real = w.real * v.real - w.imag * v.imag;
    t.imag = w.real * v.imag + w.imag * v.real;

    data[i].real = u.real + t.real;
    data[i].imag = u.imag + t.imag;
    data[j].real = u.real - t.real;
    data[j].imag = u.imag - t.imag;
}

// Host wrapper for inverse FFT
extern "C"
void ifft_gpu(GpuComplex *d_data, size_t N) {
    size_t logN = 0;
    { size_t tmp = N; while (tmp > 1) { tmp >>= 1; ++logN; } }

    // Bit-reversal permutation (same as forward FFT)
    {
        const int THREADS = 256;
        int blocks = (int)((N + THREADS - 1) / THREADS);
        bit_reversal_permute<<<blocks, THREADS>>>(d_data, N, logN);
        cudaDeviceSynchronize();
    }

    // Run each stage with ifft_stage_kernel
    for (size_t s = 1; s <= logN; ++s) {
        size_t total_pairs = N >> 1;
        const int THREADS = 256;
        int blocks2 = (int)((total_pairs + THREADS - 1) / THREADS);
        ifft_stage_kernel<<<blocks2, THREADS>>>(d_data, N, s);
        cudaDeviceSynchronize();
    }

    // Divide every element by N to complete the inverse transform
    const int THREADS2 = 256;
    int blocks3 = (int)((N + THREADS2 - 1) / THREADS2);
    scale_kernel<<<blocks3, THREADS2>>>(d_data, N, 1.0f / (float)N);
    cudaDeviceSynchronize();
}

// Kernel to scale each element by a scalar (1/N)
__global__
void scale_kernel(GpuComplex *data, size_t N, float invN) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i].real *= invN;
    data[i].imag *= invN;
}
