/*
 * Radix-2 Cooley-Tukey FFT (decimation in time) in CUDA.
 *
 * Pipeline for an N-point in-place transform (N a power of two):
 *
 *   1. Bit-reversal permutation                 fft_bitrev_*_kernel (1 launch)
 *      staged through shared memory as a tiled transpose so that both the
 *      loads and the stores are coalesced; see the comment on
 *      fft_bitrev_tiled_kernel for why the naive scatter is so expensive.
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

/* Ablation switches. Each one puts a single pre-optimization behaviour back so
 * that the cost of the corresponding change can be measured rather than
 * asserted; `make ablate` builds one binary per switch and tabulates them.
 * None of these are defined in a normal build. See docs/profiling.md.
 *
 *   FFT_ABLATE_NO_FUSION   every butterfly stage gets its own global launch
 *   FFT_ABLATE_SYNC        cudaDeviceSynchronize() between stages
 *   FFT_ABLATE_SCALAR      4-byte real/imag loads instead of 8-byte float2
 *   FFT_ABLATE_SINCOS      sinf/cosf instead of the SFU's __sincosf
 *   FFT_ABLATE_BLOCK128    128-thread blocks
 *   FFT_ABLATE_SCATTER     element-at-a-time bit reversal instead of the tiled
 *                          transpose (the old kernel is still in the tree as
 *                          the small-n path, so this one costs nothing to keep)
 */

#define FFT_TILE    1024u  /* elements per block in the shared-memory stages */
#ifdef FFT_ABLATE_BLOCK128
#define FFT_THREADS 128u
#else
#define FFT_THREADS 256u   /* block size for the elementwise kernels */
#endif
#define FFT_PI      3.14159265358979323846f

/* Bit-reversal tiling: 32 elements is 256 B of float2, one full coalesced
 * transaction, so the permutation is done on 32x32 tiles. FFT_BR_BITS is
 * log2 of that edge; the +1 column of padding keeps the transposed shared
 * reads off a single bank. */
#define FFT_BR_BITS 5u
#define FFT_BR_EDGE (1u << FFT_BR_BITS)
#define FFT_BR_ROWS 8u    /* rows of the tile handled per thread-block pass */

#ifdef FFT_ABLATE_SYNC
#define FFT_ABLATE_BARRIER() cudaDeviceSynchronize()
#else
#define FFT_ABLATE_BARRIER() ((void)0)
#endif

#ifdef FFT_ABLATE_SCATTER
#define FFT_ABLATE_SCATTER_ON 1
#else
#define FFT_ABLATE_SCATTER_ON 0
#endif

__device__ __forceinline__ float2 cplx_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* w = e^(dir * -2*pi*i * pos / m); dir = +1 forward, -1 inverse. */
__device__ __forceinline__ float2 twiddle(int dir, unsigned pos, unsigned m) {
    const float a = (float)dir * -2.0f * FFT_PI * (float)pos / (float)m;
#ifdef FFT_ABLATE_SINCOS
    return make_float2(cosf(a), sinf(a));
#else
    float s, c;
    __sincosf(a, &s, &c);
    return make_float2(c, s);
#endif
}

/* Reverse the low `bits` bits of x. __brev reverses all 32, so shift the
 * result down; bits == 0 would make that a shift by 32, which is undefined. */
__device__ __forceinline__ unsigned brev_bits(unsigned x, unsigned bits) {
    return bits ? (__brev(x) >> (32u - bits)) : 0u;
}

/* Element-at-a-time bit reversal, used only for n < FFT_BR_EDGE^2.
 * Thread i reads data[i] coalesced but data[__brev(i)] at a stride of n/32
 * or worse, so each 8-byte element costs a whole 32-byte sector in both
 * directions. That is fine for a tile that fits in L2 and ruinous past it. */
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

/* Coalesced bit reversal, for n >= FFT_BR_EDGE^2.
 *
 * Split the index into three fields, hi and lo being FFT_BR_BITS wide:
 *
 *     i = [ hi | mid | lo ]      rev(i) = [ rev(lo) | rev(mid) | rev(hi) ]
 *
 * Reversing the whole word therefore swaps the two edge fields and reverses
 * each in place. Holding `mid` fixed leaves a 32x32 tile indexed by (hi, lo)
 * whose image is the tile at mid' = rev(mid), transposed with both axes bit
 * reversed. Along a row of the source, lo runs over 32 consecutive elements
 * (256 B, one transaction); down a column of the destination, rev(hi) does
 * the same. Staging the tile in shared memory gets both.
 *
 * Tiles pair up -- mid and rev(mid) exchange contents -- so one block owns
 * the pair, loads both before storing either, and the permutation stays in
 * place with no scratch buffer. Blocks with rev(mid) < mid exit immediately;
 * a tile that is its own image (rev(mid) == mid) is permuted alone. */
static __global__ void fft_bitrev_tiled_kernel(float2 *data, unsigned logn) {
    __shared__ float2 ta[FFT_BR_EDGE][FFT_BR_EDGE + 1];
    __shared__ float2 tb[FFT_BR_EDGE][FFT_BR_EDGE + 1];

    const unsigned mid   = blockIdx.x;
    const unsigned mid_r = brev_bits(mid, logn - 2u * FFT_BR_BITS);
    if (mid_r < mid) return;
    const bool paired = (mid_r != mid);

    /* Distance between consecutive values of hi, i.e. one full row. */
    const unsigned row   = 1u << (logn - FFT_BR_BITS);
    const unsigned baseA = mid   << FFT_BR_BITS;
    const unsigned baseB = mid_r << FFT_BR_BITS;
    const unsigned tx    = threadIdx.x;

    for (unsigned hi = threadIdx.y; hi < FFT_BR_EDGE; hi += blockDim.y) {
        ta[hi][tx] = data[hi * row + baseA + tx];
        if (paired) tb[hi][tx] = data[hi * row + baseB + tx];
    }
    __syncthreads();

    /* Thread tx owns destination offset tx, so it supplies the element whose
     * hi field reverses to tx; the row index v likewise reverses to lo. */
    const unsigned rtx = brev_bits(tx, FFT_BR_BITS);
    for (unsigned v = threadIdx.y; v < FFT_BR_EDGE; v += blockDim.y) {
        const unsigned rv = brev_bits(v, FFT_BR_BITS);
        data[v * row + baseB + tx] = ta[rtx][rv];
        if (paired) data[v * row + baseA + tx] = tb[rtx][rv];
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

#ifdef FFT_ABLATE_SCALAR
/* The access pattern from before the float2 change: real and imaginary parts
 * fetched as two independent 4-byte loads. The parameter is float* rather than
 * float2* deliberately -- with only 4-byte alignment provable, nvcc cannot
 * merge the pair back into one 8-byte access and quietly undo the ablation. */
static __global__ void fft_global_kernel(float *f, unsigned pairs,
                                         unsigned half, int dir) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;

    unsigned i, pos;
    butterfly_idx(p, half, &i, &pos);
    const unsigned j = i + half;

    const float2 w = twiddle(dir, pos, half << 1);
    const float ur = f[2u * i], ui = f[2u * i + 1u];
    const float vr = f[2u * j], vi = f[2u * j + 1u];
    const float tr = w.x * vr - w.y * vi;
    const float ti = w.x * vi + w.y * vr;
    f[2u * i]      = ur + tr;
    f[2u * i + 1u] = ui + ti;
    f[2u * j]      = ur - tr;
    f[2u * j + 1u] = ui - ti;
}
#define FFT_GLOBAL_ARG(d) reinterpret_cast<float *>(d)
#else
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
#define FFT_GLOBAL_ARG(d) (d)
#endif

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
    if (logn >= 2u * FFT_BR_BITS && !FFT_ABLATE_SCATTER_ON) {
        const dim3 br_block(FFT_BR_EDGE, FFT_BR_ROWS);
        fft_bitrev_tiled_kernel<<<n >> (2u * FFT_BR_BITS), br_block>>>(d, logn);
    } else {
        const unsigned blocks_n = (n + FFT_THREADS - 1) / FFT_THREADS;
        fft_bitrev_kernel<<<blocks_n, FFT_THREADS>>>(d, n, 32 - logn);
    }
    FFT_RANGE_POP();
    FFT_ABLATE_BARRIER();

#ifdef FFT_ABLATE_NO_FUSION
    /* No shared-memory tile: start the global loop at m = 2 so that every
     * butterfly stage gets its own launch and its own DRAM round trip. */
    const unsigned tile = 1u;
#else
    FFT_RANGE_PUSH("shared stages");
    const unsigned tile = n < FFT_TILE ? n : FFT_TILE;
    fft_shared_kernel<<<n / tile, tile / 2, tile * sizeof(float2)>>>(d, tile, dir);
    FFT_RANGE_POP();
    FFT_ABLATE_BARRIER();
#endif

    FFT_RANGE_PUSH("global stages");
    const unsigned pairs = n >> 1;
    const unsigned blocks_p = (pairs + FFT_THREADS - 1) / FFT_THREADS;
    for (unsigned m = tile << 1; m != 0 && m <= n; m <<= 1) {
        fft_global_kernel<<<blocks_p, FFT_THREADS>>>(FFT_GLOBAL_ARG(d), pairs,
                                                     m >> 1, dir);
        FFT_ABLATE_BARRIER();
    }
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
