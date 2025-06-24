/*
 * Radix-2 Cooley-Tukey FFT (decimation in time) in CUDA.
 *
 * Pipeline for an N-point in-place transform (N a power of two):
 *
 *   1. Bit-reversal permutation                 fft_bitrev_kernel   (1 launch)
 *   2. First log2(FFT_TILE) butterfly stages    fft_shared_kernel   (1 launch)
 *      fused in one kernel: each block stages a contiguous tile of
 *      FFT_TILE elements through shared memory, so those stages never
 *      touch DRAM between butterflies.
 *   3. Remaining stages (stride >= FFT_TILE)    fft_global_kernel   (1 launch each)
 *      operate on global memory with fully coalesced float2 accesses:
 *      consecutive threads read/write consecutive 8-byte elements.
 *
 * All kernels are enqueued on the default stream with no intermediate
 * host synchronization; stream ordering alone enforces stage order.
 * Twiddle factors come from the SFU via __sincosf.
 *
 * The NVTX ranges below are host-side and therefore measure enqueue cost,
 * not kernel duration -- which is exactly what makes them useful in the
 * Nsight Systems timeline: they line the launch of each phase up against
 * the CUDA HW row underneath it, so launch overhead and gaps between
 * stages are visible. They compile away unless FFT_NVTX is defined.
 */
#include "fft.h"
#include "fft_profile.h"
#include <cuda_runtime.h>

#define FFT_TILE    1024u  /* elements per block in the shared-memory stages */
#define FFT_THREADS 256u   /* block size for the elementwise kernels */
#define FFT_PI      3.14159265358979323846f

__device__ __forceinline__ float2 cplx_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* w = e^(dir * -2*pi*i * pos / m); dir = +1 forward, -1 inverse. */
__device__ __forceinline__ float2 twiddle(int dir, unsigned pos, unsigned m) {
    float s, c;
    __sincosf((float)dir * -2.0f * FFT_PI * (float)pos / (float)m, &s, &c);
    return make_float2(c, s);
}

static __global__ void fft_bitrev_kernel(float2 *data, unsigned n, unsigned shift) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned r = __brev(i) >> shift;
    if (r > i) {
        const float2 t = data[i];
        data[i] = data[r];
        data[r] = t;
    }
}

/* Butterfly index math shared by the tiled and global-memory stages.
 * For pair index p in a stage with half-span `half` (m = 2*half):
 *   pos = p % half            position within the group
 *   i   = (p / half) * m + pos  upper element, j = i + half lower element */
__device__ __forceinline__ void butterfly_idx(unsigned p, unsigned half,
                                              unsigned *i, unsigned *pos) {
    *pos = p & (half - 1);
    *i   = ((p & ~(half - 1)) << 1) | *pos;
}

/* Runs stages m = 2, 4, ..., tile entirely in shared memory.
 * Each block owns one contiguous tile; loads and stores are coalesced. */
static __global__ void fft_shared_kernel(float2 *data, unsigned tile, int dir) {
    extern __shared__ float2 s_tile[];
    const unsigned base = blockIdx.x * tile;

    for (unsigned k = threadIdx.x; k < tile; k += blockDim.x)
        s_tile[k] = data[base + k];
    __syncthreads();

    for (unsigned m = 2; m <= tile; m <<= 1) {
        const unsigned half = m >> 1;
        for (unsigned p = threadIdx.x; p < (tile >> 1); p += blockDim.x) {
            unsigned i, pos;
            butterfly_idx(p, half, &i, &pos);
            const unsigned j = i + half;

            const float2 w = twiddle(dir, pos, m);
            const float2 u = s_tile[i];
            const float2 t = cplx_mul(w, s_tile[j]);
            s_tile[i] = make_float2(u.x + t.x, u.y + t.y);
            s_tile[j] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }

    for (unsigned k = threadIdx.x; k < tile; k += blockDim.x)
        data[base + k] = s_tile[k];
}

/* One butterfly stage in global memory, for strides >= FFT_TILE.
 * half >= FFT_TILE >> warp size, so consecutive threads touch consecutive
 * elements in both halves of each group: every access is coalesced. */
static __global__ void fft_global_kernel(float2 *data, unsigned pairs,
                                         unsigned half, int dir) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;

    unsigned i, pos;
    butterfly_idx(p, half, &i, &pos);
    const unsigned j = i + half;

    const float2 w = twiddle(dir, pos, half << 1);
    const float2 u = data[i];
    const float2 t = cplx_mul(w, data[j]);
    data[i] = make_float2(u.x + t.x, u.y + t.y);
    data[j] = make_float2(u.x - t.x, u.y - t.y);
}

static __global__ void fft_scale_kernel(float2 *data, unsigned n, float s) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i].x *= s;
    data[i].y *= s;
}

static void fft_run(float2 *d, size_t N, int dir) {
    const unsigned n = (unsigned)N;
    if (n < 2) return;

    unsigned logn = 0;
    while ((1u << logn) < n) ++logn;

    FFT_RANGE_PUSH("bitrev");
    const unsigned blocks_n = (n + FFT_THREADS - 1) / FFT_THREADS;
    fft_bitrev_kernel<<<blocks_n, FFT_THREADS>>>(d, n, 32 - logn);
    FFT_RANGE_POP();

    FFT_RANGE_PUSH("shared stages");
    const unsigned tile = n < FFT_TILE ? n : FFT_TILE;
    fft_shared_kernel<<<n / tile, tile / 2, tile * sizeof(float2)>>>(d, tile, dir);
    FFT_RANGE_POP();

    FFT_RANGE_PUSH("global stages");
    const unsigned pairs = n >> 1;
    const unsigned blocks_p = (pairs + FFT_THREADS - 1) / FFT_THREADS;
    for (unsigned m = tile << 1; m != 0 && m <= n; m <<= 1)
        fft_global_kernel<<<blocks_p, FFT_THREADS>>>(d, pairs, m >> 1, dir);
    FFT_RANGE_POP();
}

extern "C" int gpu_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

extern "C" void fft_gpu(GpuComplex *d_data, size_t N) {
    FFT_RANGE_PUSH("fft_gpu");
    fft_run(reinterpret_cast<float2 *>(d_data), N, +1);
    FFT_RANGE_POP();
}

extern "C" void ifft_gpu(GpuComplex *d_data, size_t N) {
    FFT_RANGE_PUSH("ifft_gpu");
    float2 *d = reinterpret_cast<float2 *>(d_data);
    fft_run(d, N, -1);

    const unsigned n = (unsigned)N;
    if (n >= 2) {
        const unsigned blocks = (n + FFT_THREADS - 1) / FFT_THREADS;
        fft_scale_kernel<<<blocks, FFT_THREADS>>>(d, n, 1.0f / (float)n);
    }
    FFT_RANGE_POP();
}
