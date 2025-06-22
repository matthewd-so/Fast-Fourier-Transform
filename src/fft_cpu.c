/*
 * Single-threaded radix-2 Cooley-Tukey FFT (decimation in time).
 * Serves as the CPU baseline for the benchmark; same algorithm the GPU
 * kernels implement. Twiddle factors use a double-precision recurrence
 * so the accumulated rotation stays accurate at large N.
 */
#include "fft.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void fft_cpu(GpuComplex *data, size_t N) {
    size_t logN = 0;
    while (((size_t)1 << logN) < N)
        ++logN;

    /* Bit-reversal permutation */
    for (size_t i = 0; i < N; ++i) {
        size_t rev = 0, x = i;
        for (size_t j = 0; j < logN; ++j) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        if (rev > i) {
            GpuComplex tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    /* Butterfly stages */
    for (size_t s = 1; s <= logN; ++s) {
        const size_t m = (size_t)1 << s;
        const size_t half = m >> 1;
        const double theta = -2.0 * M_PI / (double)m;
        const double wmr = cos(theta), wmi = sin(theta);

        for (size_t k = 0; k < N; k += m) {
            double wr = 1.0, wi = 0.0;
            for (size_t j = 0; j < half; ++j) {
                const GpuComplex v = data[k + j + half];
                const GpuComplex u = data[k + j];
                const float tr = (float)(wr * v.real - wi * v.imag);
                const float ti = (float)(wr * v.imag + wi * v.real);

                data[k + j].real = u.real + tr;
                data[k + j].imag = u.imag + ti;
                data[k + j + half].real = u.real - tr;
                data[k + j + half].imag = u.imag - ti;

                const double nwr = wr * wmr - wi * wmi;
                wi = wr * wmi + wi * wmr;
                wr = nwr;
            }
        }
    }
}
