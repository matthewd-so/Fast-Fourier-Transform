/*
 * Benchmark: this repo's radix-2 FFT vs cuFFT vs the single-threaded CPU
 * baseline, on the same input.
 *
 *   ./bench [N] [iters]     (defaults: N = 2^20, iters = 50)
 *
 * Protocol per GPU implementation: 3 untimed warmup runs, then `iters`
 * timed runs. The working buffer is restored from a pristine copy before
 * every run (untimed); CUDA events bracket only the transform itself.
 * Throughput uses the standard 5 * N * log2(N) FLOP count.
 */
#include "fft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult res_ = (call);                                            \
        if (res_ != CUFFT_SUCCESS) {                                          \
            fprintf(stderr, "%s:%d: cuFFT error %d\n", __FILE__, __LINE__,    \
                    (int)res_);                                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Relative L2 error of a against reference b. */
static double rel_error(const GpuComplex *a, const GpuComplex *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dr = (double)a[i].real - b[i].real;
        const double di = (double)a[i].imag - b[i].imag;
        num += dr * dr + di * di;
        den += (double)b[i].real * b[i].real + (double)b[i].imag * b[i].imag;
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

typedef void (*transform_fn)(GpuComplex *d_data, size_t n, void *ctx);

static void run_custom(GpuComplex *d_data, size_t n, void *) {
    fft_gpu(d_data, n);
}

static void run_cufft(GpuComplex *d_data, size_t n, void *ctx) {
    (void)n;
    cufftHandle plan = *(cufftHandle *)ctx;
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_data,
                             (cufftComplex *)d_data, CUFFT_FORWARD));
}

/* Average GPU milliseconds per transform under the warmup/restore protocol. */
static float time_transform(transform_fn fn, void *ctx, GpuComplex *d_work,
                            const GpuComplex *d_ref, size_t n, int iters) {
    const size_t bytes = n * sizeof(GpuComplex);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_work, d_ref, bytes, cudaMemcpyDeviceToDevice));
        fn(d_work, n, ctx);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float total = 0.0f;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_work, d_ref, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        fn(d_work, n, ctx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / (float)iters;
}

int main(int argc, char **argv) {
    const size_t N = argc > 1 ? (size_t)strtoull(argv[1], NULL, 0) : (size_t)1 << 20;
    const int iters = argc > 2 ? atoi(argv[2]) : 50;

    if (N < 2 || (N & (N - 1)) != 0) {
        fprintf(stderr, "N must be a power of two >= 2 (got %zu)\n", N);
        return EXIT_FAILURE;
    }
    if (!gpu_available()) {
        fprintf(stderr, "No CUDA device available.\n");
        return EXIT_FAILURE;
    }

    const size_t bytes = N * sizeof(GpuComplex);
    const double log2n = log2((double)N);
    const double flops = 5.0 * (double)N * log2n;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s | N = %zu (2^%.0f) | %d timed iterations\n\n",
           prop.name, N, log2n, iters);

    /* Deterministic complex white-noise input. */
    GpuComplex *h_in = (GpuComplex *)malloc(bytes);
    GpuComplex *h_a = (GpuComplex *)malloc(bytes);
    GpuComplex *h_b = (GpuComplex *)malloc(bytes);
    if (!h_in || !h_a || !h_b) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_in[i].real = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        h_in[i].imag = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
    }

    /* CPU baseline: best of 3 runs. */
    double cpu_ms = 1e30;
    for (int i = 0; i < 3; ++i) {
        for (size_t k = 0; k < N; ++k) h_a[k] = h_in[k];
        const double t0 = wall_ms();
        fft_cpu(h_a, N);
        const double t1 = wall_ms();
        if (t1 - t0 < cpu_ms) cpu_ms = t1 - t0;
    }

    GpuComplex *d_ref = NULL, *d_work = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_ref, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_work, bytes));
    CUDA_CHECK(cudaMemcpy(d_ref, h_in, bytes, cudaMemcpyHostToDevice));

    /* This repo's kernel. */
    const float custom_ms = time_transform(run_custom, NULL, d_work, d_ref, N, iters);
    CUDA_CHECK(cudaMemcpy(h_a, d_work, bytes, cudaMemcpyDeviceToHost));

    /* cuFFT. */
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, (int)N, CUFFT_C2C, 1));
    const float cufft_ms = time_transform(run_cufft, &plan, d_work, d_ref, N, iters);
    CUDA_CHECK(cudaMemcpy(h_b, d_work, bytes, cudaMemcpyDeviceToHost));
    CUFFT_CHECK(cufftDestroy(plan));

    /* Accuracy: against cuFFT, and forward->inverse round trip. */
    const double err_vs_cufft = rel_error(h_a, h_b, N);
    CUDA_CHECK(cudaMemcpy(d_work, d_ref, bytes, cudaMemcpyDeviceToDevice));
    fft_gpu(d_work, N);
    ifft_gpu(d_work, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_a, d_work, bytes, cudaMemcpyDeviceToHost));
    const double err_roundtrip = rel_error(h_a, h_in, N);

    printf("%-24s %10.3f ms   %8.1f GFLOP/s\n", "CPU radix-2 (1 thread)",
           cpu_ms, flops / (cpu_ms * 1e6));
    printf("%-24s %10.3f ms   %8.1f GFLOP/s   %.1fx vs CPU\n", "This kernel",
           custom_ms, flops / (custom_ms * 1e6), cpu_ms / custom_ms);
    printf("%-24s %10.3f ms   %8.1f GFLOP/s   this kernel = %.1f%% of cuFFT\n",
           "cuFFT", cufft_ms, flops / (cufft_ms * 1e6),
           100.0 * cufft_ms / custom_ms);
    printf("\nRelative L2 error vs cuFFT: %.3e\n", err_vs_cufft);
    printf("Round-trip (ifft(fft(x)) vs x) error: %.3e\n", err_roundtrip);

    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_work));
    free(h_in);
    free(h_a);
    free(h_b);
    return EXIT_SUCCESS;
}
