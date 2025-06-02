#ifndef PHYSICS_H
#define PHYSICS_H

struct Particle {
    float x, y;
    float vx, vy;
};
__global__ void updateParticlesKernel(Particle* d_particles, int N, float dt);
void initParticlesHost(Particle* h_particles, int N);

#endif
