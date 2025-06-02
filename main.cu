#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#include "physics.h"
#include "utils.h"

int main() {
    const int    N  = 1'000'000;
    const float  dt = 0.01f;

    //Allocate host memory for N Particles
    Particle *h_particles = (Particle*) malloc(N * sizeof(Particle));
    if (!h_particles) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    srand(42);
    initParticlesHost(h_particles, N);

    // 4) Allocate device memory for N Particles
    Particle *d_particles;
    size_t    size = N * sizeof(Particle);
    CHECK_CUDA(cudaMalloc(&d_particles, size));

    CHECK_CUDA(cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // Record the start event, launch the kernel, then record stop event
    CHECK_CUDA(cudaEventRecord(startEvent));
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, N, dt);
    CHECK_CUDA(cudaEventRecord(stopEvent));

    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    printf("GPU: Updated %d particles in %f ms (%.6f s)\n",
           N, milliseconds, milliseconds / 1000.0f);

    CHECK_CUDA(cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost));

    cudaFree(d_particles);
    free(h_particles);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
