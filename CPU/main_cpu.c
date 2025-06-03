#include "../fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int main(void) {
    const size_t N = 1 << 20; 
    const float freq = 50.0f;

    // Allocate a buffer of N complex samples and fill it with a sine wave
    float _Complex *cpu_data = malloc(N * sizeof(float _Complex));
    if (!cpu_data) {
        fprintf(stderr, "Could not allocate CPU buffer\n");
        return 1;
    }
    for (size_t i = 0; i < N; ++i) {
        float t = (float)i / (float)N;
        cpu_data[i] = sinf(2.0f * M_PI * freq * t) + 0.0f * I;
    }

    // Time the CPU FFT
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fft_cpu(cpu_data, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("CPU FFT completed in %.3f ms\n", elapsed_ms);

    // Print the first 5 frequency-bin magnitudes
    printf("First 5 FFT bins (magnitude):\n");
    for (int i = 0; i < 5; ++i) {
        float mag = cabsf(cpu_data[i]);
        printf("  Bin %d: %.5f\n", i, mag);
    }

    free(cpu_data);
    return 0;
}
