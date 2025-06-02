#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(err) \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

#endif
