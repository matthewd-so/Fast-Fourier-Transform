/*
 * Demo: transform a 2^20-sample sine wave and locate its spectral peak.
 *
 * Built with nvcc (default) it runs on the GPU when one is present and
 * falls back to the CPU otherwise. Built with -DFFT_CPU_ONLY (make fft_cpu)
 * it compiles without any CUDA dependency.
 */
#include "fft.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef FFT_CPU_ONLY
#include <cuda_runtime.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    const size_t N = (size_t)1 << 20;
    const float freq = 50.0f; /* cycles across the N-sample window */

    GpuComplex *x = (GpuComplex *)malloc(N * sizeof *x);
    if (!x) {
        fprintf(stderr, "error: could not allocate %zu samples\n", N);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) {
        x[i].real = sinf(2.0f * (float)M_PI * freq * (float)i / (float)N);
        x[i].imag = 0.0f;
    }

    int ran_on_gpu = 0;

#ifndef FFT_CPU_ONLY
    if (gpu_available()) {
        printf("CUDA device found. Running GPU FFT...\n");

        GpuComplex *d_data = NULL;
        const size_t bytes = N * sizeof *x;
        cudaError_t err = cudaMalloc((void **)&d_data, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            free(x);
            return EXIT_FAILURE;
        }
        cudaMemcpy(d_data, x, bytes, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        fft_gpu(d_data, N);
        cudaEventRecord(stop);

        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess || (err = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "GPU FFT failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_data);
            free(x);
            return EXIT_FAILURE;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU FFT completed in %.3f ms\n", ms);

        cudaMemcpy(x, d_data, bytes, cudaMemcpyDeviceToHost);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        ran_on_gpu = 1;
    }
#endif

    if (!ran_on_gpu) {
        printf("Running CPU FFT...\n");

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fft_cpu(x, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                          (t1.tv_nsec - t0.tv_nsec) / 1e6;
        printf("CPU FFT completed in %.3f ms\n", ms);
    }

    /* A pure sine at `freq` cycles should peak at bin 50 with |X| = N/2. */
    size_t peak = 0;
    float peak_mag = 0.0f;
    for (size_t k = 0; k < N / 2; ++k) {
        const float mag = sqrtf(x[k].real * x[k].real + x[k].imag * x[k].imag);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak = k;
        }
    }
    printf("Spectral peak: bin %zu, |X| = %.1f (expected bin %.0f, |X| = %.1f)\n",
           peak, peak_mag, freq, (float)N / 2.0f);

    free(x);
    return EXIT_SUCCESS;
}
