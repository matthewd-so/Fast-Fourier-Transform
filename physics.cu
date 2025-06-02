#include "physics.h"
#include <cstdlib>

void initParticlesHost(Particle* h_particles, int N) {
    for (int i = 0; i < N; ++i) {
        h_particles[i].x  = (float)rand() / RAND_MAX * 100.0f;
        h_particles[i].y  = (float)rand() / RAND_MAX * 100.0f;
        h_particles[i].vx = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        h_particles[i].vy = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    }
}

__global__
void updateParticlesKernel(Particle* d_particles, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float ax = 0.0f;
    const float ay = -9.8f;

    float vx = d_particles[idx].vx;
    float vy = d_particles[idx].vy;
    float x  = d_particles[idx].x;
    float y  = d_particles[idx].y;

    vx += ax * dt;
    vy += ay * dt;

    x += vx * dt;
    y += vy * dt;

    d_particles[idx].vx = vx;
    d_particles[idx].vy = vy;
    d_particles[idx].x  = x;
    d_particles[idx].y  = y;
}
