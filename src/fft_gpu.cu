#include "fft.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define THREADS 256

__device__
static GpuComplex cplx_mul(GpuComplex a, GpuComplex b) {
    GpuComplex c;
    c.real = a.real * b.real - a.imag * b.imag;
    c.imag = a.real * b.imag + a.imag * b.real;
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

// One butterfly stage. Forward and inverse differ only in the sign of the
// twiddle angle, so dir = +1 (forward) / -1 (inverse) selects between them.
__global__
void fft_stage_kernel(GpuComplex *data, size_t N, size_t stage, int dir) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = 1U << stage;
    size_t half = m >> 1;
    size_t total_pairs = N >> 1;

    if (idx >= total_pairs) return;

    size_t group = idx / half;
    size_t pos   = idx % half;

    size_t i = group * m + pos;
    size_t j = i + half;

    float angle = (float)dir * -2.0f * M_PI * ((float)pos) / ((float)m);
    GpuComplex w = {cosf(angle), sinf(angle)};

    GpuComplex u = data[i];
    GpuComplex t = cplx_mul(w, data[j]);

    data[i].real = u.real + t.real;
    data[i].imag = u.imag + t.imag;
    data[j].real = u.real - t.real;
    data[j].imag = u.imag - t.imag;
}

__global__
void scale_kernel(GpuComplex *data, size_t N, float s) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i].real *= s;
    data[i].imag *= s;
}

// Every kernel goes on the default stream, so stage k cannot start before
// stage k-1 retires: stream ordering already gives us the barrier we need.
// The caller synchronizes once, when it wants the results back.
static void fft_run(GpuComplex *d_data, size_t N, int dir) {
    // Compute log2(N)
    size_t logN = 0;
    {
        size_t tmp = N;
        while (tmp > 1) {
            tmp >>= 1;
            ++logN;
        }
    }

    int blocks = (int)((N + THREADS - 1) / THREADS);
    bit_reversal_permute<<<blocks, THREADS>>>(d_data, N, logN);

    size_t total_pairs = N >> 1;
    int blocks2 = (int)((total_pairs + THREADS - 1) / THREADS);
    for (size_t s = 1; s <= logN; ++s)
        fft_stage_kernel<<<blocks2, THREADS>>>(d_data, N, s, dir);
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
    fft_run(d_data, N, +1);
}

extern "C"
void ifft_gpu(GpuComplex *d_data, size_t N) {
    fft_run(d_data, N, -1);

    // Divide every element by N to complete the inverse transform
    int blocks = (int)((N + THREADS - 1) / THREADS);
    scale_kernel<<<blocks, THREADS>>>(d_data, N, 1.0f / (float)N);
}
