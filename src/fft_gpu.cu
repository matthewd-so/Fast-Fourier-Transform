#include "fft.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__host__ __device__
static GpuComplex gpu_exp(float theta) {
    GpuComplex c;
    c.real = cosf(theta);
    c.imag = sinf(theta);
    return c;
}

__global__
void bit_reversal_permute(GpuComplex *data, size_t N, size_t logN) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Compute bit-reversed index 'rev' of i
    size_t rev = 0;
    size_t x = i;
    for (size_t j = 0; j < logN; ++j) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (rev > i) {
        // swap data[i] and data[rev]
        GpuComplex tmp = data[i];
        data[i] = data[rev];
        data[rev] = tmp;
    }
}

__global__
void fft_stage_kernel(GpuComplex *data, size_t N, size_t stage) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = 1U << stage;       
    size_t half = m >> 1; 
    size_t total_pairs = N >> 1;

    if (idx >= total_pairs) return;

    size_t group = idx / half;
    size_t pos   = idx % half;

    size_t i = group * m + pos;
    size_t j = i + half;

    float angle = -2.0f * M_PI * ((float)pos) / ((float)m);
    GpuComplex w = gpu_exp(angle);

    GpuComplex u = data[i];
    GpuComplex v = data[j];

    GpuComplex t;
    t.real = w.real * v.real - w.imag * v.imag;
    t.imag = w.real * v.imag + w.imag * v.real;

    data[i].real = u.real + t.real;
    data[i].imag = u.imag + t.imag;
    data[j].real = u.real - t.real;
    data[j].imag = u.imag - t.imag;
}

extern "C"
int gpu_available(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return 0;
    }
    return 1;
}

extern "C"
void fft_gpu(GpuComplex *d_data, size_t N) {
    // Compute log2(N)
    size_t logN = 0;
    {
        size_t tmp = N;
        while (tmp > 1) {
            tmp >>= 1;
            ++logN;
        }
    }

    const int THREADS = 256;
    int blocks = (int)((N + THREADS - 1) / THREADS);
    bit_reversal_permute<<<blocks, THREADS>>>(d_data, N, logN);
    cudaDeviceSynchronize();

    for (size_t s = 1; s <= logN; ++s) {
        size_t total_pairs = N >> 1; 
        int blocks2 = (int)((total_pairs + THREADS - 1) / THREADS);
        fft_stage_kernel<<<blocks2, THREADS>>>(d_data, N, s);
        cudaDeviceSynchronize();
    }
}
