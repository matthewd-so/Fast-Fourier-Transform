#include "../fft.h"
#include <complex.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// In-place radix-2 Cooley-Tukey FFT on an array of float _Complex of length N.
void fft_cpu(float _Complex *data, size_t N) {
    size_t logN = 0;
    {
        size_t tmp = N;
        while (tmp > 1) {
            tmp >>= 1;
            ++logN;
        }
    }

    // 2) Bit-reversal permutation
    for (size_t i = 0; i < N; ++i) {
        size_t rev = 0;
        size_t x = i;
        for (size_t j = 0; j < logN; ++j) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        if (rev > i) {
            float _Complex tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    // 3) Main Cooley-Tukey loop: for each stage s = 1 .. logN
    for (size_t s = 1; s <= logN; ++s) {
        size_t m = 1U << s;       
        size_t half = m >> 1;     
        float theta = -2.0f * M_PI / (float)m;
        float _Complex wm = cosf(theta) + sinf(theta) * I;

        for (size_t k = 0; k < N; k += m) {
            float _Complex w = 1.0f + 0.0f * I;
            for (size_t j = 0; j < half; ++j) {
                float _Complex t = w * data[k + j + half];
                float _Complex u = data[k + j];
                data[k + j]        = u + t;
                data[k + j + half] = u - t;
                w *= wm; 
            }
        }
    }
}
