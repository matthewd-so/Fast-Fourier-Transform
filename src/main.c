#include "fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int main(void) {
    const size_t N = 1 << 20; 
    const float freq = 50.0f;

    // Allocate and fill the input buffer on the host (CPU).
    float _Complex *h_input = (float _Complex *)malloc(N * sizeof(float _Complex));
    if (!h_input) {
        fprintf(stderr, "Error: could not allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) {
        float t = (float)i / (float)N;
        h_input[i] = sinf(2.0f * M_PI * freq * t) + 0.0f * I;
    }

    // Check if a CUDA device is available.
    if (gpu_available()) {
        printf("CUDA device found. Running GPU FFT...\n");

        // 1) Allocate device memory (N GpuComplex floats).
        GpuComplex *d_data = NULL;
        size_t bytes = N * sizeof(GpuComplex);
        cudaError_t err = cudaMalloc((void **)&d_data, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            free(h_input);
            return EXIT_FAILURE;
        }

        // 2) Copy from h_input (float _Complex) → an intermediate GpuComplex array → d_data.
        GpuComplex *h_temp = (GpuComplex *)malloc(bytes);
        if (!h_temp) {
            fprintf(stderr, "Error: could not allocate staging memory\n");
            cudaFree(d_data);
            free(h_input);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < N; ++i) {
            h_temp[i].real = crealf(h_input[i]);
            h_temp[i].imag = cimagf(h_input[i]);
        }
        cudaMemcpy(d_data, h_temp, bytes, cudaMemcpyHostToDevice);

        // 3) Launch and time the GPU FFT using CUDA events.
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        fft_gpu(d_data, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("GPU FFT completed in %.3f ms\n", ms);

        // 4) Copy results back to host (into h_temp) and print first 5 bins.
        cudaMemcpy(h_temp, d_data, bytes, cudaMemcpyDeviceToHost);
        printf("First 5 FFT bins (magnitude):\n");
        for (int i = 0; i < 5; ++i) {
            float re = h_temp[i].real;
            float im = h_temp[i].imag;
            float mag = sqrtf(re * re + im * im);
            printf("  Bin %d: %.5f\n", i, mag);
        }

        // Cleanup GPU resources
        free(h_temp);
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else {
        // No CUDA device → fall back to CPU FFT.
        printf("No CUDA device detected. Running CPU FFT...\n");

        // Copy input into a separate buffer for CPU FFT (so h_input remains intact).
        float _Complex *cpu_data = (float _Complex *)malloc(N * sizeof(float _Complex));
        if (!cpu_data) {
            fprintf(stderr, "Error: could not allocate CPU buffer\n");
            free(h_input);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < N; ++i) {
            cpu_data[i] = h_input[i];
        }

        // Time the CPU FFT using clock_gettime().
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fft_cpu(cpu_data, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                            (t1.tv_nsec - t0.tv_nsec) / 1e6;
        printf("CPU FFT completed in %.3f ms\n", elapsed_ms);

        printf("First 5 FFT bins (magnitude):\n");
        for (int i = 0; i < 5; ++i) {
            float mag = cabsf(cpu_data[i]);
            printf("  Bin %d: %.5f\n", i, mag);
        }

        free(cpu_data);
    }

    free(h_input);
    return 0;
}
