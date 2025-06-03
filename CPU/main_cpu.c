#include "../fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

int main(void) {
    const size_t N = 1 << 20;  // 1,048,576 samples

    // Seed the random number generator for white noise
    srand((unsigned)time(NULL));

    // Allocate an array of N complex samples and generate white noise
    float _Complex *cpu_data = malloc(N * sizeof(float _Complex));
    if (!cpu_data) {
        fprintf(stderr, "Error: could not allocate CPU buffer\n");
        return 1;
    }
    for (size_t i = 0; i < N; ++i) {
        float r = (float)rand() / (float)RAND_MAX; 
        r = 2.0f * r - 1.0f;                  
        cpu_data[i] = r + 0.0f * I;
    }

    // Run and time the in‐place CPU FFT
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fft_cpu(cpu_data, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("CPU FFT completed in %.3f ms\n", elapsed_ms);

    // Print the magnitudes of the first 100 FFT bins
    printf("First 100 FFT bins (magnitude):\n");
    for (int i = 0; i < 100; ++i) {
        float mag = cabsf(cpu_data[i]);
        printf("  Bin %3d: %.5f\n", i, mag);
    }

    free(cpu_data);
    return 0;
}
